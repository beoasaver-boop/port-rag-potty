"""RAG portable · cliente de embeddings contra Ollama (bge-m3)."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

import httpx

from . import config

log = logging.getLogger("rag.embeddings")


class EmbeddingError(RuntimeError):
    """Error generando embeddings contra Ollama."""


async def _embed_one(client: httpx.AsyncClient, text: str) -> list[float]:
    """Llama a /api/embeddings para un único texto (Ollama)."""
    payload = {"model": config.EMBED_MODEL, "prompt": text}
    try:
        r = await client.post(
            f"{config.OLLAMA_HOST.rstrip('/')}/api/embeddings",
            json=payload,
            timeout=60.0,
        )
    except httpx.HTTPError as e:
        raise EmbeddingError(f"Ollama no responde en {config.OLLAMA_HOST}: {e}") from e

    if r.status_code != 200:
        raise EmbeddingError(
            f"Ollama devolvió {r.status_code}: {r.text[:200]}"
        )
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise EmbeddingError(f"Respuesta inesperada: {data!r}")
    return emb


async def embed_texts(texts: Iterable[str], concurrency: int = 4) -> list[list[float]]:
    """Genera embeddings en paralelo limitado.

    Chroma acepta listas de embeddings, pero Ollama no expone batch; hacemos
    peticiones concurrentes con un semáforo para no saturar al servidor.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[list[float] | None] = [None] * len(texts)  # type: ignore[list-item]

    async def _one(i: int, txt: str) -> None:
        async with sem:
            for attempt in range(3):
                try:
                    results[i] = await _embed_one(client, txt)
                    return
                except EmbeddingError as e:
                    if attempt == 2:
                        raise
                    log.warning("retry %d: %s", attempt + 1, e)
                    await asyncio.sleep(0.6 * (attempt + 1))

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[_one(i, t) for i, t in enumerate(texts)])

    return [r for r in results if r is not None]  # type: ignore[misc]


def embed_texts_sync(texts: Iterable[str], concurrency: int = 4) -> list[list[float]]:
    """Wrapper síncrono para usar desde el indexer."""
    return asyncio.run(embed_texts(texts, concurrency=concurrency))


async def health_check() -> dict:
    """Comprueba que Ollama responde y el modelo está disponible."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{config.OLLAMA_HOST.rstrip('/')}/api/tags")
            if r.status_code != 200:
                return {"ok": False, "reason": f"tags {r.status_code}"}
            tags = r.json().get("models", [])
            has_model = any(m.get("name", "").startswith(config.EMBED_MODEL) for m in tags)
            return {
                "ok": has_model,
                "ollama_host": config.OLLAMA_HOST,
                "embed_model": config.EMBED_MODEL,
                "model_loaded": has_model,
                "available_models": [m.get("name") for m in tags],
            }
        except httpx.HTTPError as e:
            return {"ok": False, "reason": str(e), "ollama_host": config.OLLAMA_HOST}
