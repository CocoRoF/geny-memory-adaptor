"""geny-executor bridge — duck-typed handles, NO import of geny_executor.

geny-executor resolves memory backends structurally (Protocols), so this
module ships classes whose shapes match the executor's ``VectorHandle`` and a
retriever-style facade — without depending on the executor package. When the
Geny side wires Synapse in, it can either:

* hand ``SynapseVectorHandle`` to ``FileMemoryProvider(vector_store=...)`` —
  drop-in replacement for the API-embedding vector layer, keeping everything
  else identical; or
* register a full provider that delegates STM/LTM/Notes to the existing file
  provider and routes ``retrieve()`` through ``SynapseRetriever``.

Both entry points are plain sync calls wrapped async — every operation is
CPU-bound milliseconds, so there is nothing to await.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .engine import SearchHit, SynapseMemory


class SynapseVectorHandle:
    """Structural match for geny-executor's ``VectorHandle`` Protocol:
    ``descriptor / index / index_batch / search / reindex / remove``."""

    def __init__(self, memory: SynapseMemory, *, name: str = "synapse") -> None:
        self._m = memory
        self._name = name

    @property
    def descriptor(self) -> Dict[str, Any]:
        return {
            "backend": self._name,
            "model": "synapse-hash-static",
            "dimension": self._m.cfg.dim,
            "local": True,
            "api_calls": 0,
        }

    async def index(self, doc_id: str, text: str,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        md = metadata or {}
        self._m.index(
            doc_id, text,
            title=str(md.get("title") or ""),
            kind=str(md.get("kind") or md.get("category") or "note"),
            tags=list(md.get("tags") or ()),
            links=list(md.get("links") or ()),
            importance=float(md.get("importance") or 1.0),
            pinned=bool(md.get("pinned")),
            teacher_vec=md.get("teacher_vec"),
            teacher_model=str(md.get("teacher_model") or ""),
        )

    async def index_batch(self, docs: Sequence[Dict[str, Any]]) -> int:
        for d in docs:
            await self.index(d["id"], d.get("text") or "", d.get("metadata"))
        return len(docs)

    async def search(self, query: str, *, top_k: int = 8,
                     score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        hits = self._m.search(query, top_k=top_k)
        return [
            {
                "id": h.id, "score": h.score, "title": h.title, "kind": h.kind,
                "sources": h.sources, "query_token": h.query_token,
            }
            for h in hits if h.score >= score_threshold
        ]

    async def remove(self, doc_id: str) -> None:
        self._m.remove(doc_id)

    async def reindex(self) -> int:
        # Synapse indexes are derived + incremental; distill() is the only
        # batch maintenance and doubles as "reindex with a better model".
        metrics = self._m.distill()
        return int(metrics.get("pairs", 0))

    async def fetch_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        node = self._m.store.get_node(doc_id)
        return dict(node) if node else None


class SynapseRetriever:
    """Retrieval facade shaped like a host-side memory retriever: one call
    returns ranked hits ready for prompt injection, plus the feedback hook the
    host calls when it learns which memories were actually used."""

    def __init__(self, memory: SynapseMemory) -> None:
        self._m = memory

    async def retrieve(self, query: str, *, top_k: int = 8,
                       kinds: Optional[Sequence[str]] = None) -> List[SearchHit]:
        return self._m.search(query, top_k=top_k, kinds=kinds)

    async def feedback(self, query_token: str, used_ids: Sequence[str],
                       *, label_src: str = "implicit") -> Dict[str, float]:
        return self._m.feedback(query_token, used_ids=used_ids, label_src=label_src)
