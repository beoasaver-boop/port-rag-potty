"""RAG portable · indexer (chunk + embed + upsert en Chroma).

Flujo:
  1. Cargar manifest.json (mtimes + ids por archivo)
  2. Walk del proyecto → por cada archivo:
     - Si mtime cambió o es nuevo: re-chunkear
     - Si no: saltar
  3. Embeddings vía Ollama (bge-m3) en lotes
  4. Upsert en ChromaDB (delete-by-file antes si re-indexamos)
  5. Re-construir grafo (rápido)
  6. Guardar manifest + graph.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import chromadb

from . import config
from . import chunker
from . import embeddings
from . import graph as graph_mod

log = logging.getLogger("rag.indexer")


# ── Manifest (estado de indexación) ────────────────────────────────────────
class Manifest:
    """Por archivo: mtime + ids de chunks (para borrado selectivo)."""

    def __init__(self) -> None:
        self.files: dict[str, dict] = {}  # rel -> {mtime, chunk_ids, n}

    def to_dict(self) -> dict:
        return self.files

    @classmethod
    def load(cls) -> "Manifest":
        if config.MANIFEST_FILE.exists():
            try:
                return cls.from_dict(json.loads(config.MANIFEST_FILE.read_text("utf-8")))
            except json.JSONDecodeError:
                pass
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        m = cls()
        m.files = d
        return m

    def save(self) -> None:
        config.MANIFEST_FILE.write_text(
            json.dumps(self.files, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_mtime(self, rel: str) -> float:
        return self.files.get(rel, {}).get("mtime", 0.0)

    def set_file(self, rel: str, mtime: float, chunk_ids: list[str], n: int) -> None:
        self.files[rel] = {"mtime": mtime, "chunk_ids": chunk_ids, "n": n}

    def remove_file(self, rel: str) -> None:
        self.files.pop(rel, None)


# ── Cliente Chroma ─────────────────────────────────────────────────────────
def _get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_dim": config.EMBED_DIM},
    )


# ── Reconciliación manifest ↔ Chroma ───────────────────────────────────────
def _present_ids(collection, ids: list[str], batch: int = 256) -> set[str]:
    """Subconjunto de `ids` que existe realmente en Chroma."""
    present: set[str] = set()
    for i in range(0, len(ids), batch):
        part = ids[i : i + batch]
        try:
            res = collection.get(ids=part, include=[])
            present.update(res.get("ids") or [])
        except Exception as e:  # noqa: BLE001
            # Si la lectura falla, asumimos presentes para no re-embeber todo.
            log.warning("reconciliación: get(%d ids) falló (%s)", len(part), e)
            present.update(part)
    return present


def reconcile(collection, manifest: "Manifest") -> list[str]:
    """Archivos cuyo manifest declara chunks que YA NO están en Chroma.

    Auto-cura el estado envenenado: si una indexación borró chunks y falló antes
    de reinsertarlos, el manifest sigue diciendo "al día" y el reindex
    incremental los saltaría para siempre.
    """
    stale: list[str] = []
    for rel, info in manifest.files.items():
        ids = list(info.get("chunk_ids") or [])
        if not ids:
            stale.append(rel)
            continue
        missing = len(ids) - len(_present_ids(collection, ids))
        if missing:
            log.warning(
                "reconciliación: %s declara %d chunks y faltan %d en Chroma → se reindexa",
                rel, len(ids), missing,
            )
            stale.append(rel)
    return stale


def _delete_ids(collection, ids: list[str], batch: int = 256) -> int:
    """Borra ids de Chroma; devuelve cuántos se intentaron borrar."""
    if not ids:
        return 0
    deleted = 0
    for i in range(0, len(ids), batch):
        part = ids[i : i + batch]
        try:
            collection.delete(ids=part)
            deleted += len(part)
        except Exception as e:  # noqa: BLE001
            log.warning("delete chunks falló (%d ids): %s", len(part), e)
    return deleted


# ── Run indexación ─────────────────────────────────────────────────────────
def run_index(
    *,
    project_root: Path | None = None,
    force: bool = False,
    progress_cb=None,
) -> dict:
    """Indexar (incremental o forzado). Devuelve estadísticas.

    Invariantes:
      1. Nunca se borra un chunk viejo antes de tener el nuevo dentro de Chroma.
      2. El manifest sólo registra ids que se upsertearon de verdad.
      3. Si un archivo queda incompleto, se marca para reintento (mtime=0),
         en lugar de quedar "al día" con chunks perdidos.
    """
    project_root = project_root or config.PROJECT_ROOT
    started = time.time()

    manifest = Manifest.load()
    collection = _get_collection()

    files_indexed = 0
    chunks_added = 0
    chunks_removed = 0
    failures = 0

    # 0) Reconciliación: archivos cuyo manifest declara chunks que ya no existen
    stale_files: set[str] = set() if force else set(reconcile(collection, manifest))
    if stale_files:
        log.info("reconciliación: %d archivo(s) a reindexar", len(stale_files))

    # Plan de trabajo: {rel, mtime, new_ids, old_ids}
    planned: list[dict[str, Any]] = []
    # Buffer para procesar en batch (embed → upsert)
    pending_chunks: list[chunker.Chunk] = []

    for p in chunker.iter_project_files(project_root):
        rel = str(p.relative_to(project_root))
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue

        prev = manifest.files.get(rel, {})
        prev_mtime = prev.get("mtime", 0.0)
        prev_ids = list(prev.get("chunk_ids") or [])

        if (
            not force
            and rel not in stale_files
            and mtime == prev_mtime
            and prev_ids
        ):
            continue

        new_chunks = chunker.chunk_file(p, project_root)
        if not new_chunks:
            # El archivo ya no produce chunks: retirar los viejos del índice
            chunks_removed += _delete_ids(collection, prev_ids)
            manifest.remove_file(rel)
            continue

        # OJO: no se borra nada todavía. Primero chunk, luego embed, luego upsert
        # y sólo al final se retiran los ids viejos que no se hayan reutilizado.
        planned.append({
            "rel": rel,
            "mtime": mtime,
            "new_ids": [c.id for c in new_chunks],
            "old_ids": prev_ids,
        })
        pending_chunks.extend(new_chunks)
        files_indexed += 1

        if progress_cb:
            progress_cb(f"chunked: {rel} ({len(new_chunks)})")

    # 1) Embeddings (todavía sin tocar Chroma)
    ok_ids: set[str] = set()
    if pending_chunks:
        texts = [c.embed_text() for c in pending_chunks]
        if progress_cb:
            progress_cb(f"embedding {len(texts)} chunks con {config.EMBED_MODEL}…")

        # Intentar en lote primero
        vecs: list[list[float] | None] = [None] * len(pending_chunks)  # type: ignore[list-item]
        BATCH = 32
        for i in range(0, len(texts), BATCH):
            j = i + BATCH
            batch_texts = texts[i:j]
            try:
                batch_vecs = embeddings.embed_texts_sync(batch_texts)
                for k, v in enumerate(batch_vecs):
                    vecs[i + k] = v
            except embeddings.EmbeddingError as e:
                log.warning("batch %d-%d falló (%s); reintentando 1 a 1", i, j, e)
                # Reintentar uno a uno para encontrar el problemático
                for k, txt in enumerate(batch_texts):
                    try:
                        vecs[i + k] = embeddings.embed_texts_sync([txt])[0]
                    except embeddings.EmbeddingError as e2:
                        failures += 1
                        log.error("chunk %d falló: %s", i + k, e2)

        ok_indices = [i for i, v in enumerate(vecs) if v is not None]

        # 1b) Fallo total de embeddings: NO se borra nada (el índice queda
        # intacto) y se marca lo planificado con mtime=0 para reintentar, de
        # modo que el reindex incremental no lo salte como "al día".
        if not ok_indices:
            for item in planned:
                manifest.set_file(item["rel"], 0.0, item["old_ids"], len(item["old_ids"]))
            manifest.save()
            return {
                "ok": False,
                "error": "Ningún chunk pudo embedderse (índice intacto; se reintentará)",
                "files_indexed": 0,
                "files_pending_retry": len(planned),
                "elapsed_s": round(time.time() - started, 2),
            }

        # 2) Upsert de lo que sí se embebió
        ids = [pending_chunks[i].id for i in ok_indices]
        ok_vecs = [vecs[i] for i in ok_indices]  # type: ignore[misc]
        ok_texts = [texts[i] for i in ok_indices]
        metas = [pending_chunks[i].to_meta() for i in ok_indices]
        ok_ids = set(ids)

        # Chroma acepta lotes grandes; metemos todos de una
        UPSERT_BATCH = 256
        for i in range(0, len(ids), UPSERT_BATCH):
            j = i + UPSERT_BATCH
            collection.upsert(
                ids=ids[i:j],
                embeddings=ok_vecs[i:j],
                documents=ok_texts[i:j],
                metadatas=metas[i:j],
            )
        chunks_added = len(ids)
        if failures:
            log.warning("saltados por fallo: %d", failures)

        # 3) Ya con los nuevos dentro, retirar los viejos no reutilizados
        for item in planned:
            to_delete = [i for i in item["old_ids"] if i not in ok_ids]
            chunks_removed += _delete_ids(collection, to_delete)

    # 4) Manifest: sólo ids realmente presentes en Chroma. Un archivo con chunks
    #    fallidos se queda con mtime=0 para que la próxima pasada lo reintente.
    incomplete = 0
    for item in planned:
        new_ok = [i for i in item["new_ids"] if i in ok_ids]
        if len(new_ok) == len(item["new_ids"]):
            manifest.set_file(item["rel"], item["mtime"], new_ok, len(new_ok))
        else:
            incomplete += 1
            log.warning(
                "incompleto: %s (%d/%d chunks) → se reintentará en el próximo reindex",
                item["rel"], len(new_ok), len(item["new_ids"]),
            )
            manifest.set_file(item["rel"], 0.0, new_ok, len(new_ok))

    # Eliminar del manifest los archivos que ya no existen
    existing = {str(p.relative_to(project_root)) for p in chunker.iter_project_files(project_root)}
    stale = [r for r in manifest.files if r not in existing]
    for rel in stale:
        prev = manifest.files.pop(rel, {})
        chunks_removed += _delete_ids(collection, list(prev.get("chunk_ids") or []))

    manifest.save()

    # Re-construir grafo
    if progress_cb:
        progress_cb("construyendo grafo…")
    g = graph_mod.build_graph(project_root)
    g.save(config.GRAPH_FILE)

    return {
        "ok": True,
        "files_indexed": files_indexed,
        "files_reconciled": len(stale_files),
        "files_incomplete": incomplete,
        "chunks_failed": failures,
        "chunks_added": chunks_added,
        "chunks_removed": chunks_removed,
        "categories": len(g.categories),
        "tokens": len(g.tokens),
        "files_in_graph": len(g.files),
        "edges": len(g.edges),
        "elapsed_s": round(time.time() - started, 2),
    }


# ── Stats rápidos ─────────────────────────────────────────────────────────
def collection_stats() -> dict:
    """Snapshot del estado actual de la colección (rápido)."""
    try:
        c = _get_collection()
        count = c.count()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    return {"ok": True, "count": count, "model": config.EMBED_MODEL}


def raw_query(where: dict | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """Inspección rápida de chunks (sin embedding) — útil para debug."""
    c = _get_collection()
    res = c.get(where=where, limit=limit, include=["metadatas", "documents"])
    out = []
    for i, _id in enumerate(res["ids"]):
        out.append({
            "id": _id,
            "doc": res["documents"][i],
            "meta": res["metadatas"][i],
        })
    return out
