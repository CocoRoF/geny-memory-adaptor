"""SynapseMemory — the public engine: index / search / feedback / distill.

One object per vault. All state lives in `<path>` (SQLite) + `<path>.emb.npz`
(embedding table). Zero network calls; every operation is CPU-milliseconds.

    mem = SynapseMemory.open(path="vault/synapse.db")     # or from_env()
    mem.index("note-1", "본문…", title="제목", tags=["게임"], links=["note-0"])
    hits = mem.search("리듬게임 판정", top_k=8)
    mem.feedback(hits[0].query_token, used_ids=["note-1"])  # ← 온라인 학습
    mem.distill()                                           # teacher가 있을 때만
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .bm25 import bm25_scores, term_frequencies, top_n
from .config import SynapseConfig
from .embedder import HashEmbedder, pack_vec, unpack_vec
from .graph import (
    build_type_adjacency,
    derive_knn_edges,
    derive_tag_edges,
    ppr_features,
    reinforce_coaccess,
)
from .ranker import FEATURES, OnlineRanker
from .store import EDGE_COACCESS, EDGE_KNN, EDGE_LINK, EDGE_TAG, Store
from .tokenizer import tokenize

_KIND_PRIOR = {"fact": 1.0, "insight": 0.8, "note": 0.5, "digest": 0.4, "turn": 0.2}


@dataclass
class SearchHit:
    id: str
    score: float
    title: str = ""
    kind: str = "note"
    features: Dict[str, float] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)  # which retrievers hit
    #: Opaque token identifying this query for feedback().
    query_token: str = ""


class SynapseMemory:
    def __init__(self, config: Optional[SynapseConfig] = None, **overrides: Any) -> None:
        cfg = config or SynapseConfig()
        if overrides:
            cfg = SynapseConfig(**{**cfg.__dict__, **{k: v for k, v in overrides.items()
                                                      if k in SynapseConfig.__dataclass_fields__}})
        self.cfg = cfg
        self.store = Store(cfg.path)
        self._emb_path = cfg.emb_path or (
            "" if cfg.path == ":memory:" else cfg.path + ".emb.npz")
        self.embedder = self._load_embedder()
        self.ranker = self._load_ranker()
        # Incremental caches — the difference between O(N) and O(N²) bulk
        # indexing, and between ~8ms and ~50ms queries at a few thousand
        # nodes. All derived from SQLite; safe to drop at any time.
        self._vec_cache: Optional[Dict[str, np.ndarray]] = None
        self._vec_matrix: Optional[tuple] = None          # (ids, np.ndarray)
        self._doclen_cache: Optional[Dict[str, int]] = None
        self._tag_cache: Optional[Dict[str, List[str]]] = None
        self._adj_cache: Dict[int, Optional[list]] = {}
        #: query_token → {"hash":…, "features": {node_id: np.ndarray}, "shown": [ids]}
        self._recent_queries: Dict[str, Dict[str, Any]] = {}
        self._rng = random.Random(cfg.seed)

    # ── construction helpers ─────────────────────────────────────────
    @classmethod
    def open(cls, path: str = "synapse.db", **overrides: Any) -> "SynapseMemory":
        return cls(SynapseConfig(path=path), **overrides)

    @classmethod
    def from_env(cls, *, dotenv: Optional[str] = None, **overrides: Any) -> "SynapseMemory":
        """Configure via GMA_* env vars (optionally loading a .env first)."""
        return cls(SynapseConfig.from_env(dotenv=dotenv, **overrides))

    def _load_embedder(self) -> HashEmbedder:
        if self._emb_path and os.path.isfile(self._emb_path):
            try:
                return HashEmbedder.load(self._emb_path, seed=self.cfg.seed,
                                         char_ngrams=self.cfg.char_ngrams)
            except Exception:
                pass
        blob = self.store.get_param("embedder")
        if blob:
            try:
                return HashEmbedder.loads(blob, seed=self.cfg.seed,
                                          char_ngrams=self.cfg.char_ngrams)
            except Exception:
                pass
        return HashEmbedder(self.cfg.vocab_size, self.cfg.dim, seed=self.cfg.seed,
                            char_ngrams=self.cfg.char_ngrams)

    def _load_ranker(self) -> OnlineRanker:
        blob = self.store.get_param("ranker")
        if blob:
            try:
                return OnlineRanker.loads(blob)
            except Exception:
                pass
        return OnlineRanker(hidden=self.cfg.hidden, lr=self.cfg.lr, l2=self.cfg.l2,
                            blend_min_events=self.cfg.blend_min_events)

    # ── write path ───────────────────────────────────────────────────
    def index(
        self,
        node_id: str,
        text: str,
        *,
        title: str = "",
        kind: str = "note",
        tags: Sequence[str] = (),
        links: Sequence[str] = (),
        importance: float = 1.0,
        pinned: bool = False,
        updated_at: Optional[float] = None,
        teacher_vec: Optional[Sequence[float]] = None,
        teacher_model: str = "",
    ) -> None:
        """Index (or re-index) one memory. Idempotent by *node_id*.

        *teacher_vec* is an ALREADY-COMPUTED higher-quality embedding the
        caller happens to have (e.g. a stored API embedding) — used only as a
        distillation label, never required.
        """
        body = f"{title}\n{text}" if title else text
        tokens = tokenize(body, char_ngrams=self.cfg.char_ngrams,
                          limit=self.cfg.max_doc_tokens)
        self.store.upsert_node(
            node_id, kind=kind, title=title, tags=tags, text_len=len(tokens),
            updated_at=updated_at or time.time(), pinned=pinned, importance=importance)
        self.store.replace_postings(node_id, term_frequencies(tokens))
        vec = self.embedder.embed(body, limit=self.cfg.max_doc_tokens)
        self.store.put_vector(node_id, pack_vec(vec), self.cfg.dim)
        # Incremental cache maintenance (never a full invalidate on write).
        if self._vec_cache is not None:
            self._vec_cache[node_id] = vec
        self._vec_matrix = None
        if self._doclen_cache is not None:
            self._doclen_cache[node_id] = len(tokens)
        if self._tag_cache is not None:
            for t in tags:
                members = self._tag_cache.setdefault(t, [])
                if node_id not in members:
                    members.append(node_id)
        # Edges: explicit links (bidirectional), tags, semantic kNN.
        self.store.replace_edges_from(
            node_id, EDGE_LINK, [(dst, 1.0) for dst in links])
        if links:
            self.store.upsert_edges([(dst, node_id, EDGE_LINK, 1.0) for dst in links])
        self.store.replace_edges_from(
            node_id, EDGE_TAG,
            derive_tag_edges(self._tags_map(), self._n_docs(), node_id, tags,
                             fanout=self.cfg.tag_fanout))
        vectors = self._vectors()
        self.store.replace_edges_from(
            node_id, EDGE_KNN,
            derive_knn_edges(vec, vectors, node_id,
                             k=self.cfg.knn_edges, min_sim=self.cfg.knn_min_sim))
        self._adj_cache.clear()
        if teacher_vec is not None:
            t = np.asarray(teacher_vec, dtype=np.float32)
            self.store.put_teacher(node_id, teacher_model, pack_vec(t), int(t.shape[0]))
        self._save_text_for_distill(node_id, body)

    def remove(self, node_id: str) -> None:
        self.store.remove_node(node_id)
        if self._vec_cache is not None:
            self._vec_cache.pop(node_id, None)
        self._vec_matrix = None
        if self._doclen_cache is not None:
            self._doclen_cache.pop(node_id, None)
        self._tag_cache = None
        self._adj_cache.clear()

    # ── read path ────────────────────────────────────────────────────
    def search(self, query: str, *, top_k: Optional[int] = None,
               kinds: Optional[Sequence[str]] = None) -> List[SearchHit]:
        top_k = top_k or self.cfg.top_k
        q_tokens = tokenize(query, char_ngrams=self.cfg.char_ngrams,
                            limit=self.cfg.max_query_tokens)
        now = time.time()

        # ① seeds — BM25 ∪ cosine, fused by RRF.
        bm25 = bm25_scores(self.store, q_tokens, doc_lens=self._doclens(),
                           k1=self.cfg.bm25_k1, b=self.cfg.bm25_b)
        q_vec = self.embedder.embed(query, limit=self.cfg.max_query_tokens)
        ids, matrix = self._vector_matrix()
        cos: Dict[str, float] = {}
        if ids:
            sims = matrix @ q_vec
            # Only the top slice matters; argpartition keeps this O(N).
            k = min(len(ids), max(self.cfg.vector_seed_k * 4, 64))
            for j in np.argpartition(-sims, k - 1)[:k]:
                cos[ids[int(j)]] = float(sims[int(j)])
        bm_top = top_n(bm25, self.cfg.bm25_seed_k)
        cos_top = sorted(cos, key=lambda k: -cos[k])[: self.cfg.vector_seed_k]
        rrf: Dict[str, float] = {}
        for rank, nid in enumerate(bm_top):
            rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (60 + rank)
        for rank, nid in enumerate(cos_top):
            rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (60 + rank)
        seeds = {nid: score for nid, score in rrf.items()}
        if not seeds:
            return []

        # ② graph expansion — per-type PPR from the seed distribution.
        ppr = ppr_features(self._adjacencies(), seeds, alpha=self.cfg.ppr_alpha,
                           iters=self.cfg.ppr_iters)
        candidates = set(seeds)
        expansion_pool: Dict[str, float] = {}
        for etype_scores in ppr.values():
            for nid, s in etype_scores.items():
                if nid not in candidates:
                    expansion_pool[nid] = max(expansion_pool.get(nid, 0.0), s)
        for nid in sorted(expansion_pool, key=lambda k: -expansion_pool[k])[: self.cfg.graph_expand_k]:
            candidates.add(nid)

        # ③ features + ranking.
        meta = {n["id"]: n for n in self.store.nodes(candidates)}
        q_words = set(w for w in q_tokens if len(w) > 1)
        feats: Dict[str, np.ndarray] = {}
        scored: List[SearchHit] = []
        for nid in candidates:
            node = meta.get(nid)
            if node is None:
                continue
            if kinds and node["kind"] not in kinds:
                continue
            age_days = max(0.0, (now - (node["updated_at"] or now)) / 86400.0)
            title_words = set(tokenize(node["title"], char_ngrams=(), limit=32))
            x = np.array([
                bm25.get(nid, 0.0),
                cos.get(nid, 0.0),
                rrf.get(nid, 0.0),
                ppr[EDGE_LINK].get(nid, 0.0),
                ppr[EDGE_TAG].get(nid, 0.0),
                ppr[EDGE_KNN].get(nid, 0.0),
                ppr[EDGE_COACCESS].get(nid, 0.0),
                1.0 / (1.0 + math.log1p(age_days)),
                math.log1p(node["access_count"]),
                node["importance"],
                1.0 if node["pinned"] else 0.0,
                1.0 if q_words & title_words else 0.0,
                _KIND_PRIOR.get(node["kind"], 0.5),
                min(1.0, node["text_len"] / 512.0),
            ], dtype=np.float32)
            self.ranker.observe(x)
            feats[nid] = x
            score = self.ranker.score(x)
            sources = []
            if nid in bm25:
                sources.append("bm25")
            if nid in cos and cos[nid] > 0:
                sources.append("vector")
            if nid not in seeds:
                sources.append("graph")
            scored.append(SearchHit(id=nid, score=score, title=node["title"],
                                    kind=node["kind"],
                                    features={f: float(v) for f, v in zip(FEATURES, x)},
                                    sources=sources))
        scored.sort(key=lambda h: -h.score)
        result = scored[:top_k]

        # ε-exploration: swap the tail slot with a random non-shown candidate.
        if (len(scored) > top_k and result and
                self._rng.random() < self.cfg.epsilon):
            result[-1] = self._rng.choice(scored[top_k:])

        # Register for feedback.
        token = hashlib.sha1(f"{query}|{now}".encode()).hexdigest()[:16]
        for h in result:
            h.query_token = token
        self._recent_queries[token] = {
            "hash": hashlib.sha1(query.encode()).hexdigest()[:16],
            "features": {h.id: feats[h.id] for h in result if h.id in feats},
            "shown": [h.id for h in result],
        }
        if len(self._recent_queries) > 64:
            self._recent_queries.pop(next(iter(self._recent_queries)))
        self.store.touch_access([h.id for h in result], ts=now)
        return result

    # ── learning path ────────────────────────────────────────────────
    def feedback(self, query_token: str, *, used_ids: Sequence[str] = (),
                 ignored_ids: Optional[Sequence[str]] = None,
                 label_src: str = "implicit") -> Dict[str, float]:
        """Report which shown memories were actually USED.

        *ignored_ids* defaults to shown-minus-used. Triggers: Hebbian
        co-access reinforcement, pairwise ranker SGD (event + small replay),
        and persists everything. Cost: microseconds-to-ms.
        """
        q = self._recent_queries.get(query_token)
        if q is None:
            return {"applied": 0.0}
        used = [i for i in used_ids if i in q["features"]]
        ignored = list(ignored_ids) if ignored_ids is not None else \
            [i for i in q["shown"] if i not in set(used_ids)]
        ignored = [i for i in ignored if i in q["features"]]

        # ① Hebbian graph learning.
        if len(used) >= 2:
            reinforce_coaccess(self.store, used, eta=self.cfg.hebb_eta,
                               decay=self.cfg.hebb_decay, prune=self.cfg.hebb_prune)
            self._adj_cache.pop(EDGE_COACCESS, None)

        # ② Ranker: event pairs + replay refresh.
        loss = 0.0
        pairs = 0
        for u in used:
            for g in ignored:
                loss += self.ranker.update_pair(q["features"][u], q["features"][g])
                pairs += 1
        for nid in used:
            self.store.add_feedback(q["hash"], nid, q["features"][nid].tobytes(),
                                    True, label_src, self.cfg.replay_cap)
        for nid in ignored:
            self.store.add_feedback(q["hash"], nid, q["features"][nid].tobytes(),
                                    False, label_src, self.cfg.replay_cap)
        pairs += self._replay(8)
        self._persist_models()
        return {"applied": 1.0, "pairs": float(pairs),
                "loss": loss / max(1, pairs), **self.ranker.stats()}

    def _replay(self, n_pairs: int) -> int:
        rows = self.store.feedback_rows(256)
        pos = [(q, np.frombuffer(f, dtype=np.float32)) for q, f, u in rows if u]
        neg = [(q, np.frombuffer(f, dtype=np.float32)) for q, f, u in rows if not u]
        by_q: Dict[str, Dict[str, list]] = {}
        for q, x in pos:
            by_q.setdefault(q, {"p": [], "n": []})["p"].append(x)
        for q, x in neg:
            by_q.setdefault(q, {"p": [], "n": []})["n"].append(x)
        eligible = [(q, d) for q, d in by_q.items() if d["p"] and d["n"]]
        done = 0
        while eligible and done < n_pairs:
            q, d = self._rng.choice(eligible)
            self.ranker.update_pair(self._rng.choice(d["p"]), self._rng.choice(d["n"]))
            done += 1
        return done

    # ── distillation ────────────────────────────────────────────────
    def distill(self, *, epochs: Optional[int] = None) -> Dict[str, float]:
        """Fit the embedding table to stored teacher vectors (if any), then
        RE-EMBED every node with the improved table. Bounded batch job —
        run at session close / idle, never on the hot path."""
        teachers = self.store.teachers()
        texts = self._distill_texts()
        pairs = []
        for node_id, _model, dim, blob in teachers:
            text = texts.get(node_id)
            if not text:
                continue
            pairs.append((text, unpack_vec(blob, dim)))
        metrics = self.embedder.distill(
            pairs, epochs=epochs or self.cfg.distill_epochs,
            lr=self.cfg.distill_lr, batch=self.cfg.distill_batch)
        if metrics.get("swapped"):
            for node_id, text in texts.items():
                vec = self.embedder.embed(text, limit=self.cfg.max_doc_tokens)
                self.store.put_vector(node_id, pack_vec(vec), self.cfg.dim)
            self._vec_cache = None
        self._persist_models()
        return metrics

    # ── misc ─────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": self.store.count_nodes(),
            "feedback_rows": self.store.feedback_count(),
            "edges": {name: len(self.store.edges_by_type(t)) for name, t in
                      (("link", EDGE_LINK), ("tag", EDGE_TAG),
                       ("knn", EDGE_KNN), ("coaccess", EDGE_COACCESS))},
            "ranker": self.ranker.stats(),
            "dim": self.cfg.dim,
        }

    def close(self) -> None:
        self._persist_models()
        self.store.close()

    def __enter__(self) -> "SynapseMemory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── internals ────────────────────────────────────────────────────
    def _vectors(self) -> Dict[str, np.ndarray]:
        if self._vec_cache is None:
            self._vec_cache = {
                nid: unpack_vec(blob, dim)
                for nid, dim, blob in self.store.all_vectors()
            }
        return self._vec_cache

    def _vector_matrix(self) -> tuple:
        """(ids, row-stacked matrix) — cached; rebuilt lazily after writes."""
        if self._vec_matrix is None:
            vectors = self._vectors()
            ids = list(vectors.keys())
            matrix = np.stack([vectors[i] for i in ids]) if ids else \
                np.zeros((0, self.cfg.dim), dtype=np.float32)
            self._vec_matrix = (ids, matrix)
        return self._vec_matrix

    def _doclens(self) -> Dict[str, int]:
        if self._doclen_cache is None:
            self._doclen_cache = self.store.doc_lens()
        return self._doclen_cache

    def _n_docs(self) -> int:
        return len(self._doclens())

    def _tags_map(self) -> Dict[str, List[str]]:
        if self._tag_cache is None:
            tag_map: Dict[str, List[str]] = {}
            for node in self.store.nodes():
                for t in node["tags"]:
                    tag_map.setdefault(t, []).append(node["id"])
            self._tag_cache = tag_map
        return self._tag_cache

    def _adjacencies(self) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        for etype in (EDGE_LINK, EDGE_TAG, EDGE_KNN, EDGE_COACCESS):
            cached = self._adj_cache.get(etype)
            if cached is None:
                cached = build_type_adjacency(
                    self.store.edges_by_type(etype), etype,
                    coaccess_decay=self.cfg.hebb_decay)
                self._adj_cache[etype] = cached
            out[etype] = cached
        return out

    def _persist_models(self) -> None:
        self.store.put_param("ranker", self.ranker.dumps())
        if self._emb_path:
            try:
                self.embedder.save(self._emb_path)
            except OSError:
                self.store.put_param("embedder", self.embedder.dumps())
        else:
            self.store.put_param("embedder", self.embedder.dumps())

    def _save_text_for_distill(self, node_id: str, body: str) -> None:
        # Distillation needs the text back; store a bounded copy in params-space.
        key = f"text:{node_id}"
        self.store.put_param(key, body[:4000].encode("utf-8"))

    def _distill_texts(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for node in self.store.nodes():
            blob = self.store.get_param(f"text:{node['id']}")
            if blob:
                out[node["id"]] = blob.decode("utf-8", "replace")
        return out
