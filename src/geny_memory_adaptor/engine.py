"""SynapseMemory — the public engine: index / search / feedback / distill.

One object per vault. All state — vectors, edges, the learned ranker AND the
embedding table — lives in the single `<path>` SQLite file. Zero network
calls; every operation is CPU-milliseconds. All public methods are thread-safe
(one re-entrant lock guards the caches, ranker, and embedder).

    mem = SynapseMemory.open(path="vault/synapse.db")     # or from_env()
    mem.index("note-1", "본문…", title="제목", tags=["게임"], links=["note-0"])
    hits = mem.search("리듬게임 판정", top_k=8)
    mem.feedback(hits[0].query_token, used_ids=["note-1"])  # ← 온라인 학습
    mem.distill()                                           # teacher가 있을 때만
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
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
from .tokenizer import lexical_tokens

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
        # One re-entrant lock guards ALL mutable engine state (caches, ranker,
        # embedder, _recent_queries, _rng). The Store has its own lock for
        # SQLite, but these Python structures are mutated by index/search/
        # feedback/distill/remove and must not interleave across threads —
        # ``check_same_thread=False`` means callers CAN hit one instance from
        # several threads (e.g. a search turn + a background feedback).
        self._lock = threading.RLock()

    # ── construction helpers ─────────────────────────────────────────
    @classmethod
    def open(cls, path: str = "synapse.db", **overrides: Any) -> "SynapseMemory":
        return cls(SynapseConfig(path=path), **overrides)

    @classmethod
    def from_env(cls, *, dotenv: Optional[str] = None, **overrides: Any) -> "SynapseMemory":
        """Configure via GMA_* env vars (optionally loading a .env first)."""
        return cls(SynapseConfig.from_env(dotenv=dotenv, **overrides))

    def _load_embedder(self) -> HashEmbedder:
        kw = dict(seed=self.cfg.seed, char_ngrams=self.cfg.char_ngrams,
                  jamo_ngrams=self.cfg.jamo_ngrams, suffix_strip=self.cfg.suffix_strip)
        blob = self.store.get_param("embedder")
        if blob:
            try:
                return HashEmbedder.loads(blob, **kw)
            except Exception:
                pass
        return HashEmbedder(self.cfg.vocab_size, self.cfg.dim, **kw)

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
        with self._lock:
            existing = self.store.get_node(node_id)  # None ⇒ fresh insert
            body = f"{title}\n{text}" if title else text
            # ── compute everything FIRST (reads + numpy), commit ONCE ──
            # LEXICAL stream → BM25 postings (words + stems + syllable bigrams
            # + cross-space bigrams). The jamo-augmented EMBEDDING stream lives
            # only inside the hash embedder.
            tok_kw = dict(char_ngrams=self.cfg.char_ngrams,
                          suffix_strip=self.cfg.suffix_strip,
                          cross_space=self.cfg.cross_space)
            tokens = lexical_tokens(body, limit=self.cfg.max_doc_tokens, **tok_kw)
            # BM25F-lite: title terms weigh title_boost× (the title already
            # appears once inside `body`, so the extra weight is title_boost−1).
            tf = term_frequencies(tokens)
            if title and self.cfg.title_boost > 1.0:
                extra = self.cfg.title_boost - 1.0
                for t in set(lexical_tokens(title, limit=64, **tok_kw)):
                    tf[t] = tf.get(t, 0.0) + extra
            vec = self.embedder.embed(body, limit=self.cfg.max_doc_tokens)
            # Edges. LINK is stored ONE-directional and symmetrized at query
            # time (build_type_adjacency), so a changed link set can't orphan a
            # reverse edge. TAG/KNN derived from current graph state.
            tag_edges = derive_tag_edges(self._tags_map(), self._n_docs(), node_id,
                                         tags, fanout=self.cfg.tag_fanout)
            knn_edges = derive_knn_edges(vec, self._vectors(), node_id,
                                         k=self.cfg.knn_edges, min_sim=self.cfg.knn_min_sim,
                                         sample_cap=self.cfg.knn_sample_cap)
            teacher = None
            if teacher_vec is not None:
                tv = np.asarray(teacher_vec, dtype=np.float32)
                teacher = (teacher_model, pack_vec(tv), int(tv.shape[0]))
            text_param = ((f"text:{node_id}", body[:self.cfg.store_text_maxlen].encode("utf-8"))
                          if self.cfg.store_text else None)

            # ── one atomic transaction: node + postings + vector + edges ──
            self.store.index_atomic(
                node_id, kind=kind, title=title, tags=tags, text_len=len(tokens),
                updated_at=updated_at or time.time(), pinned=pinned, importance=importance,
                tf=tf, vec=pack_vec(vec), dim=self.cfg.dim,
                edges=[(EDGE_LINK, [(dst, 1.0) for dst in links]),
                       (EDGE_TAG, tag_edges), (EDGE_KNN, knn_edges)],
                teacher=teacher, text_param=text_param)

            # ── cache maintenance (after the commit succeeds) ──
            if self._vec_cache is not None:
                self._vec_cache[node_id] = vec
            self._vec_matrix = None
            if self._doclen_cache is not None:
                self._doclen_cache[node_id] = len(tokens)
            # Tag cache: INCREMENTAL on both insert and re-index. On a re-index
            # remove the node from its OLD tags (read off `existing`) then add
            # the new ones — O(#tags), NOT an O(N) full-cache rebuild (which
            # made re-indexing a big corpus O(N²): 541 ms/re-index at 40k).
            if self._tag_cache is not None:
                if existing is not None:
                    for t in existing["tags"]:
                        lst = self._tag_cache.get(t)
                        if lst and node_id in lst:
                            lst.remove(node_id)
                for t in tags:
                    lst = self._tag_cache.setdefault(t, [])
                    if node_id not in lst:
                        lst.append(node_id)
            self._adj_cache.clear()

    def remove(self, node_id: str) -> None:
        with self._lock:
            self.store.remove_node(node_id)
            self.store.delete_param(f"text:{node_id}")
            if self._vec_cache is not None:
                self._vec_cache.pop(node_id, None)
            self._vec_matrix = None
            if self._doclen_cache is not None:
                self._doclen_cache.pop(node_id, None)
            self._tag_cache = None
            self._adj_cache.clear()
            # Drop any pending feedback tokens that reference this node — else
            # feedback() could reinforce co-access edges or re-insert feedback
            # rows pointing at a now-deleted node.
            for tok in list(self._recent_queries):
                if node_id in self._recent_queries[tok]["features"]:
                    del self._recent_queries[tok]

    # ── read path ────────────────────────────────────────────────────
    def search(self, query: str, *, top_k: Optional[int] = None,
               kinds: Optional[Sequence[str]] = None) -> List[SearchHit]:
        with self._lock:
            return self._search(query, top_k=top_k, kinds=kinds)

    def _search(self, query: str, *, top_k: Optional[int] = None,
                kinds: Optional[Sequence[str]] = None) -> List[SearchHit]:
        # `is None` (not truthiness) so an explicit top_k=0 means "0 results",
        # and clamp negatives to 0 rather than slicing from the tail.
        top_k = self.cfg.top_k if top_k is None else max(0, top_k)
        q_tokens = lexical_tokens(query, char_ngrams=self.cfg.char_ngrams,
                                  suffix_strip=self.cfg.suffix_strip,
                                  cross_space=self.cfg.cross_space,
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
            title_words = set(lexical_tokens(node["title"], char_ngrams=(),
                                             cross_space=False, limit=32))
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
            # NOTE: do NOT observe() here — updating the normalization stats
            # mid-ranking makes a candidate perturb its own (and later
            # candidates') z-scores, so search is non-idempotent. Score against
            # the current stats, collect the features, and fold them into the
            # running normalization AFTER ranking (below).
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

        # Fold this query's candidate features into the running normalization
        # AFTER ranking — so the scores above are idempotent w.r.t. this call.
        for x in feats.values():
            self.ranker.observe(x)

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
        with self._lock:
            return self._feedback(query_token, used_ids=used_ids,
                                  ignored_ids=ignored_ids, label_src=label_src)

    def _feedback(self, query_token: str, *, used_ids: Sequence[str] = (),
                  ignored_ids: Optional[Sequence[str]] = None,
                  label_src: str = "implicit") -> Dict[str, float]:
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

        # ② Ranker. Split by QUERY: a deterministic ~25% of queries are HELD
        # OUT — refereed for the blend gate, never trained on (and never added
        # to the replay buffer), so the gate's McNemar test is truly
        # out-of-sample. The rest train the live weights + feed replay.
        is_eval = (int(q["hash"], 16) % 4 == 0)
        loss = 0.0
        pairs = 0
        for u in used:
            for g in ignored:
                if is_eval:
                    self.ranker.referee_pair(q["features"][u], q["features"][g])
                else:
                    loss += self.ranker.update_pair(q["features"][u], q["features"][g])
                    pairs += 1
        if not is_eval:
            for nid in used:
                self.store.add_feedback(q["hash"], nid, q["features"][nid].tobytes(),
                                        True, label_src, self.cfg.replay_cap)
            for nid in ignored:
                self.store.add_feedback(q["hash"], nid, q["features"][nid].tobytes(),
                                        False, label_src, self.cfg.replay_cap)
            pairs += self._replay(8)
        self._persist_ranker()  # ranker only — the embedder is untouched here
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
    def distill(self, *, epochs: Optional[int] = None) -> Dict[str, Any]:
        """Fit the embedding table to stored teacher vectors, then ATOMICALLY
        swap the table and re-embed every stored-text node in one transaction.
        Needs ``store_text=True`` (the whole corpus must be re-embeddable) and
        enough teacher pairs. Bounded batch job — run at idle / close."""
        with self._lock:
            if not self.cfg.store_text:
                return {"trained": 0.0, "reason_no_text": 1.0}
            teachers = self.store.teachers()
            texts = self._distill_texts()
            pairs = [(texts[nid], unpack_vec(blob, dim))
                     for nid, _m, dim, blob in teachers if texts.get(nid)]
            metrics = self.embedder.distill(
                pairs, epochs=epochs or self.cfg.distill_epochs,
                lr=self.cfg.distill_lr, batch=self.cfg.distill_batch)
            candidate = metrics.pop("candidate", None)
            if candidate is not None:
                # Re-embedding must cover EVERY node, else the un-covered ones
                # keep OLD-table vectors that the NEW query embedder can't match.
                all_ids = {n["id"] for n in self.store.nodes()}
                if not all_ids.issubset(texts.keys()):
                    metrics["swapped"] = 0.0
                    metrics["reason_incomplete_text"] = 1.0
                    self._persist_ranker()
                    return metrics
                # Build the candidate on a SCRATCH embedder and swap it in ONLY
                # after the store commit succeeds. If the commit rolls back, the
                # live embedder + on-disk table + vectors all stay OLD together —
                # no in-memory/disk mismatch for the rest of the session.
                scratch = HashEmbedder(
                    self.cfg.vocab_size, self.cfg.dim, seed=self.cfg.seed,
                    char_ngrams=self.cfg.char_ngrams, jamo_ngrams=self.cfg.jamo_ngrams,
                    suffix_strip=self.cfg.suffix_strip)
                scratch.table = candidate
                rows = [(nid, self.cfg.dim,
                         pack_vec(scratch.embed(txt, limit=self.cfg.max_doc_tokens)))
                        for nid, txt in texts.items()]
                self.store.swap_embedder_and_vectors(scratch.dumps(), rows)
                self.embedder = scratch  # commit succeeded → adopt atomically
                self._vec_cache = None
                self._vec_matrix = None
            self._persist_ranker()
            return metrics

    # ── misc ─────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        with self._lock:
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
        with self._lock:
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

    def _persist_ranker(self) -> None:
        """Persist ONLY the ranker — tiny (~1 KB), safe to write every feedback."""
        self.store.put_param("ranker", self.ranker.dumps())

    def _persist_embedder(self) -> None:
        """Persist the embedding table (~fp16, tens of MB) — ONLY after
        distillation actually mutates it, never on the feedback hot path
        (writing it every feedback was a 14 s/20-call zlib bottleneck). Lives
        in the same db as the vectors so distill can swap both atomically."""
        self.store.put_param("embedder", self.embedder.dumps())

    def _persist_models(self) -> None:
        self._persist_ranker()
        self._persist_embedder()

    def get_text(self, node_id: str) -> Optional[str]:
        """The stored (bounded) body of a node, or None. Requires
        ``store_text=True``. Lets a host return the actual text alongside a
        search hit — e.g. to fill a retrieval result's ``content`` — without
        keeping the corpus in a second place."""
        with self._lock:
            blob = self.store.get_param(f"text:{node_id}")
            return blob.decode("utf-8", "replace") if blob else None

    def _save_text_for_distill(self, node_id: str, body: str) -> None:
        # Distillation needs the text back; store a bounded copy in params-space.
        key = f"text:{node_id}"
        self.store.put_param(key, body[:self.cfg.store_text_maxlen].encode("utf-8"))

    def _distill_texts(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for node in self.store.nodes():
            blob = self.store.get_param(f"text:{node['id']}")
            if blob:
                out[node["id"]] = blob.decode("utf-8", "replace")
        return out
