"""RAG portable · servidor FastAPI (genérico).

Endpoints:
  GET  /health                 → estado Ollama + colección
  GET  /methodology            → chunks 'methodology' (alma del proyecto)
  GET  /categories             → lista con resumen 1-línea
  GET  /category/{id}          → categoría + tokens + archivos + chunks
  POST /query                  → búsqueda semántica {q, k, category?, file_glob?}
  GET  /related/{kind}/{id}    → vecinos del grafo (category|file|token)
  POST /reindex                → reconstruir índice (admin)
  GET  /stats                  → estadísticas del sistema
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import chromadb
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from . import config
from . import embeddings
from . import graph as graph_mod
from . import indexer

log = logging.getLogger("rag.server")
logging.basicConfig(level=os.environ.get("RAG_LOG_LEVEL", "INFO"))

# ── Estado del reindex (un único job a la vez) ─────────────────────────────
_REINDEX_LOCK = threading.Lock()
REINDEX_STATE: dict[str, Any] = {
    "running": False,
    "status": "idle",        # idle | running | ok | error
    "force": None,
    "started_at": None,
    "finished_at": None,
    "stats": None,
    "error": None,
}


# ── Lifespan: carga Chroma + grafo al arrancar ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    app.state.graph = graph_mod.Graph.load(config.GRAPH_FILE)
    log.info("RAG server up · proyecto=%s Ollama=%s model=%s chroma=%s",
             config.PROJECT_NAME, config.OLLAMA_HOST, config.EMBED_MODEL, config.CHROMA_DIR)
    yield


app = FastAPI(
    title=f"RAG portable · {config.PROJECT_NAME}",
    version="2.0",
    description="Servicio RAG local con bge-m3 + ChromaDB + grafo ligero (portable).",
    lifespan=lifespan,
)


def _collection():
    return app.state.chroma.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_dim": config.EMBED_DIM},
    )


def _graph() -> graph_mod.Graph:
    return app.state.graph


# ── Modelos de request/response ───────────────────────────────────────────
class QueryReq(BaseModel):
    q: str = Field(..., description="Pregunta o frase a buscar.")
    k: int = Field(5, ge=1, le=20, description="Número de chunks a devolver.")
    category: str | None = Field(None, description="Filtrar por categoría del grafo.")
    file_glob: str | None = Field(None, description="Filtrar por patrón de archivo (substring).")
    include_methodology: bool = Field(True, description="Inyectar siempre chunks 'methodology'.")


class ChunkOut(BaseModel):
    id: str
    score: float | None = None
    file: str
    kind: str
    section: str
    category: str = ""
    start_line: int
    end_line: int
    snippet: str
    methodology: bool = False


class QueryResp(BaseModel):
    q: str
    k: int
    results: list[ChunkOut]
    methodology_chunks: list[ChunkOut] | None = None
    elapsed_ms: float


def _chunkout(meta: dict, doc: str | None, cid: str,
              score: float | None = None, max_chars: int = 600) -> ChunkOut:
    return ChunkOut(
        id=cid,
        score=score,
        file=meta.get("file", ""),
        kind=meta.get("kind", ""),
        section=meta.get("section", ""),
        category=meta.get("category", ""),
        start_line=meta.get("start_line", 0),
        end_line=meta.get("end_line", 0),
        snippet=(doc or "")[:max_chars],
        methodology=bool(meta.get("methodology", False)),
    )


# ── /health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    ollama = await embeddings.health_check()
    stats = indexer.collection_stats()
    g = _graph()
    return {
        "ok": True,
        "ollama": ollama,
        "collection": stats,
        "graph": {
            "categories": len(g.categories),
            "tokens": len(g.tokens),
            "files": len(g.files),
            "edges": len(g.edges),
        },
        "config": {
            "project": config.PROJECT_NAME,
            "embed_model": config.EMBED_MODEL,
            "embed_dim": config.EMBED_DIM,
            "ollama_host": config.OLLAMA_HOST,
            "chroma_dir": str(config.CHROMA_DIR),
            "collection": config.CHROMA_COLLECTION,
        },
    }


# ── /methodology ──────────────────────────────────────────────────────────
@app.get("/methodology")
async def methodology():
    """Devuelve SIEMPRE los chunks marcados como 'methodology' (alma del proyecto)."""
    col = _collection()
    res = col.get(
        where={"methodology": True},
        include=["documents", "metadatas"],
        limit=50,
    )
    out = []
    for i, cid in enumerate(res["ids"]):
        out.append(_chunkout(res["metadatas"][i], res["documents"][i], cid))
    return {"count": len(out), "chunks": out}


# ── /categories ───────────────────────────────────────────────────────────
@app.get("/categories")
async def list_categories():
    g = _graph()
    categories = []
    for cid, data in sorted(g.categories.items()):
        categories.append({
            "id": cid,
            "summary": g.category_summaries.get(cid, ""),
            "n_tokens": len(data.get("tokens", set())),
            "n_files": len(data.get("files", set())),
            "n_scripts": len(data.get("scripts", set())),
            "files": sorted(data.get("files", set()))[:6],
        })
    return {"count": len(categories), "categories": categories}


# ── /category/{id} ────────────────────────────────────────────────────────
@app.get("/category/{category_id}")
async def category_detail(category_id: str):
    """Detalle completo de una categoría: tokens, archivos, scripts + chunks."""
    g = _graph()
    c = g.categories.get(category_id)
    if not c:
        raise HTTPException(404, f"Categoría '{category_id}' no encontrada en el grafo")

    tokens = sorted(c.get("tokens", set()))
    files = sorted(c.get("files", set()))
    scripts = sorted(c.get("scripts", set()))

    col = _collection()
    res = col.get(
        where={"category": category_id},
        include=["metadatas", "documents"],
        limit=200,
    )
    chunks = []
    for i, cid in enumerate(res["ids"]):
        meta = res["metadatas"][i]
        chunks.append({
            "id": cid,
            "file": meta.get("file", ""),
            "kind": meta.get("kind", ""),
            "section": meta.get("section", ""),
            "start_line": meta.get("start_line", 0),
            "snippet": (res["documents"][i] or "")[:400],
        })

    return {
        "id": category_id,
        "summary": g.category_summaries.get(category_id, ""),
        "tokens": tokens[:60],
        "n_tokens_total": len(tokens),
        "files": files,
        "scripts": scripts,
        "chunks_sample": chunks[:30],
        "n_chunks": len(chunks),
    }


# ── /query (búsqueda semántica) ──────────────────────────────────────────
@app.post("/query", response_model=QueryResp)
async def query(req: QueryReq):
    t0 = time.time()

    try:
        vecs = await embeddings.embed_texts([req.q])
    except embeddings.EmbeddingError as e:
        raise HTTPException(503, f"No se pudo embeber la consulta: {e}")
    if not vecs:
        raise HTTPException(503, "Embeddings vacíos")
    qvec = vecs[0]

    # Construir filtro Chroma
    where: dict | None = None
    conds: list[dict] = []
    if req.category:
        conds.append({"category": req.category})
    if req.file_glob:
        files = sorted(f for f in indexer.Manifest.load().files if req.file_glob in f)
        if not files:
            raise HTTPException(
                404,
                f"Ningún archivo indexado coincide con file_glob={req.file_glob!r}. "
                "Comprueba /stats (manifest_files) o lanza POST /reindex.",
            )
        conds.append({"file": {"$in": files}})
    if len(conds) == 1:
        where = conds[0]
    elif len(conds) > 1:
        where = {"$and": conds}

    col = _collection()
    res = col.query(
        query_embeddings=[qvec],
        n_results=req.k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results: list[ChunkOut] = []
    if res.get("ids") and res["ids"][0]:
        for i, cid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            dist = res["distances"][0][i] if res.get("distances") else None
            score = round(1 - dist, 4) if dist is not None else None
            results.append(_chunkout(meta, res["documents"][0][i], cid, score=score))
            if len(results) >= req.k:
                break

    methodology_chunks: list[ChunkOut] | None = None
    if req.include_methodology:
        methodo = col.get(
            where={"methodology": True},
            include=["documents", "metadatas"],
            limit=3,
        )
        methodology_chunks = [
            _chunkout(methodo["metadatas"][i], methodo["documents"][i], cid, max_chars=500)
            for i, cid in enumerate(methodo["ids"])
        ]

    return QueryResp(
        q=req.q,
        k=req.k,
        results=results,
        methodology_chunks=methodology_chunks,
        elapsed_ms=round((time.time() - t0) * 1000, 1),
    )


# ── /related/{kind}/{id} ──────────────────────────────────────────────────
@app.get("/related/{kind}/{entity_id}")
async def related(kind: str, entity_id: str):
    """Vecinos en el grafo (1 salto)."""
    g = _graph()
    if kind not in ("category", "file", "token"):
        raise HTTPException(400, "kind ∈ {category, file, token}")
    entity = f"{kind}:{entity_id}"
    neighbours = graph_mod.related_entity(g, entity)
    return {"entity": entity, **neighbours}


# ── /reindex ──────────────────────────────────────────────────────────────
class ReindexResp(BaseModel):
    ok: bool
    message: str
    details: dict[str, Any] | None = None


@app.post("/reindex", response_model=ReindexResp)
async def reindex(force: bool = False, bg: BackgroundTasks = None):
    """Reconstruir índice (background). Un solo job a la vez; estado en /stats."""
    if not _REINDEX_LOCK.acquire(blocking=False):
        raise HTTPException(
            409,
            "Ya hay un reindex en curso. Mira /stats → reindex "
            "(running/status/started_at) y espera a que termine.",
        )

    REINDEX_STATE.update(
        running=True,
        status="running",
        force=force,
        started_at=time.time(),
        finished_at=None,
        stats=None,
        error=None,
    )

    def _run():
        try:
            stats = indexer.run_index(force=force)
            app.state.graph = graph_mod.Graph.load(config.GRAPH_FILE)
            log.info("reindex done: %s", stats)
            REINDEX_STATE.update(
                status="ok" if stats.get("ok") else "error",
                stats=stats,
                error=None if stats.get("ok") else stats.get("error"),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("reindex failed: %s", e)
            REINDEX_STATE.update(status="error", error=str(e), stats=None)
        finally:
            REINDEX_STATE.update(running=False, finished_at=time.time())
            _REINDEX_LOCK.release()

    try:
        threading.Thread(target=_run, daemon=True, name="rag-reindex").start()
    except Exception:  # noqa: BLE001
        _REINDEX_LOCK.release()
        REINDEX_STATE.update(running=False, status="error", error="no se pudo lanzar el thread")
        raise
    return ReindexResp(
        ok=True,
        message="Reindex iniciado en background. Estado en /stats → reindex.",
        details={"force": force, "collection": config.CHROMA_COLLECTION},
    )


# ── /stats ────────────────────────────────────────────────────────────────
@app.get("/stats")
async def stats():
    col = _collection()
    count = col.count()
    g = _graph()
    manifest = indexer.Manifest.load()
    declared = sum(v.get("n", 0) for v in manifest.files.values())
    return {
        "chroma_chunks": count,
        "manifest_files": len(manifest.files),
        "manifest_chunks": declared,
        "drift": count - declared,
        "reindex": dict(REINDEX_STATE),
        "graph": {
            "categories": len(g.categories),
            "tokens": len(g.tokens),
            "files": len(g.files),
            "edges": len(g.edges),
        },
    }


# ── CLI para arrancar sin uvicorn ─────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run(
        "rag.server:app",
        host=config.RAG_HOST,
        port=config.RAG_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()