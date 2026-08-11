"""Deep-agent retrieval: vector search -> Voyage rerank -> OpenAI answer + memory write.

Reuses the pipeline's active-collection vector search (`pipeline.retrieval`). Reranking and
answer synthesis degrade gracefully: if the Voyage rerank model or the OpenAI key is
unavailable, the endpoint still returns ranked source chunks.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from pipeline.clients import mongo_client, voyage_client
from pipeline.config import settings
from pipeline.config_store import get_active
from pipeline.retrieval import vector_search

_SYSTEM = (
    "You are a precise assistant answering from a MongoDB Atlas knowledge base built by a "
    "Temporal + Voyage ingestion pipeline. Answer ONLY from the provided sources. Cite sources "
    "inline as [n]. If the sources don't contain the answer, say so plainly."
)

_LOG = logging.getLogger(__name__)


def _rerank(query: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not docs:
        return docs
    try:
        res = voyage_client().rerank(
            query, [d["text"] for d in docs], model=settings.voyage_rerank_model, top_k=top_k
        )
        ordered = []
        for r in res.results:
            d = dict(docs[r.index])
            d["rerank_score"] = r.relevance_score
            ordered.append(d)
        return ordered
    except Exception:
        return docs[:top_k]  # rerank unavailable -> keep vector order


def _synthesize(query: str, docs: list[dict[str, Any]]) -> str | None:
    if not settings.openai_api_key or settings.openai_api_key.startswith("<"):
        return None
    from openai import AuthenticationError, NotFoundError, OpenAI

    context = "\n\n".join(
        f"[{i + 1}] (source: {d['source_uri']})\n{d['text']}" for i, d in enumerate(docs)
    )
    client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        client_kwargs["base_url"] = settings.openai_base_url
        if "api.openai.com" not in settings.openai_base_url.lower():
            client_kwargs["default_headers"] = {
                "Ocp-Apim-Subscription-Key": settings.openai_api_key,
                "api-key": settings.openai_api_key,
            }
            client_kwargs["default_query"] = {"subscription-key": settings.openai_api_key}
    client = OpenAI(**client_kwargs)
    try:
        resp = client.chat.completions.create(
            model=settings.answer_model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nSources:\n{context}"},
            ],
        )
        return resp.choices[0].message.content or None
    except AuthenticationError as exc:
        _LOG.warning("OpenAI authentication failed; returning sources only: %s", exc)
        return None
    except NotFoundError as exc:
        _LOG.warning(
            "OpenAI model '%s' was not found or is not accessible; returning sources only: %s",
            settings.answer_model,
            exc,
        )
        return None
    except Exception as exc:
        _LOG.warning("OpenAI synthesis failed; returning sources only: %s", exc)
        return None


def _query_hash(query: str) -> str:
    """Stable key used to deduplicate memory entries for the same question."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def _remember(query: str, answer: str | None, docs: list[dict[str, Any]], sources: list[dict[str, Any]]) -> None:
    try:
        mongo_client()[settings.mongodb_db][settings.memory_collection].update_one(
            {"query_hash": _query_hash(query)},
            {"$set": {
                "query": query,
                "answer": answer,
                "chunk_ids": [d["chunk_id"] for d in docs],
                "sources": sources,
                "model": settings.answer_model,
                "ts": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
    except Exception:
        pass  # memory write is best-effort


def _recall(query: str) -> dict[str, Any] | None:
    """Return the cached memory entry for this question, or None."""
    try:
        return mongo_client()[settings.mongodb_db][settings.memory_collection].find_one(
            {"query_hash": _query_hash(query)}
        )
    except Exception:
        return None


def ask(query: str, k: int = 10, top_k: int = 5) -> dict[str, Any]:
    """Run the deep-agent retrieval + synthesis for one query."""
    active = get_active()

    cached = _recall(query)
    if cached and cached.get("answer"):
        return {
            "query": query,
            "answer": cached["answer"],
            "answer_available": True,
            "from_memory": True,
            "active_collection": active["active_collection"],
            "model": cached.get("model", active["model"]),
            "sources": cached.get("sources", []),
        }

    hits = vector_search(query, k=k)
    ranked = _rerank(query, hits, top_k)
    answer = _synthesize(query, ranked)
    sources = [
        {
            "n": i + 1,
            "s3_uri": d["source_uri"],
            "chunk_id": d["chunk_id"],
            "score": round(float(d.get("rerank_score", d.get("score", 0.0))), 4),
            "vector_score": round(float(d.get("score", 0.0)), 4),
            "text": d["text"][:600],
        }
        for i, d in enumerate(ranked)
    ]
    _remember(query, answer, ranked, sources)

    return {
        "query": query,
        "answer": answer,
        "answer_available": answer is not None,
        "from_memory": False,
        "active_collection": active["active_collection"],
        "model": active["model"],
        "sources": sources,
    }
