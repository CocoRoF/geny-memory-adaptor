# Changelog

## [1.2.1] — 2026-07-20

Final gatekeeper review (three more adversarial passes: fix-correctness,
integration/deadlock/E2E, plus hand-verification). Ship verdict was
conditional on freezing these fixes — this is that freeze.

### Fixed
- **Blend gate was not actually out-of-sample** (the 1.2.0 fix's premise).
  The held-out split was per-CALL-ordinal, and `_replay()` fed already-trained
  pairs through the same counter — so "held-out" referee pairs were routinely
  in-sample replays, letting recurring-pair label noise open the gate.
  Rebuilt as a QUERY-LEVEL holdout: a deterministic ~25% of queries (by hash)
  go only to `referee_pair()` (never trained, never buffered), the rest train.
  Integer counts, Wilson z=3.0, and a 0.52 win-rate floor (absorbs residual
  document overlap between held-out and trained queries). Verified: moderate
  real signal → λ 0.70; pure noise → held-out win-rate 0.50, gate shut; even
  engine-level noise with document overlap stays shut.
- **distill left the in-memory embedder ahead of disk on a failed commit** —
  `self.embedder.table` was swapped BEFORE the store transaction, so a
  rollback left queries using the new table over old vectors. Now built on a
  scratch embedder and adopted only after the commit succeeds. Plus an
  `all_ids ⊆ stored-text` guard so a partial re-embed can't orphan nodes on
  the old table.
- **LINK symmetrization double-weighted reciprocated links** (A↔B counted
  twice in the row-normalized adjacency, distorting `ppr_link`). Dedup by pair
  (max weight).
- **search() was non-idempotent and could explode** — `observe()` ran during
  ranking (a candidate perturbed its own z-score) and a near-constant feature
  drove `var → 0`, swinging absolute scores ~100× on small corpora. observe()
  now runs AFTER ranking; z-scores are clipped to ±8 (contains the blow-up
  without rescaling informative features — MIRACL-ko nDCG unchanged at 0.593).

### Verified solid (adversarial, reproduced)
- No deadlock (8-thread mixed ops), no crash, no data corruption; persistence
  bit-exact for the learned ranker (mu/var/weights/counts) and edge tables;
  remove() fully purges edges/caches/feedback tokens; every degenerate input
  degrades gracefully.

## [1.2.0] — 2026-07-20

Adversarial-review hardening. Three independent reviewers (correctness,
numerical/ML-theory, persistence/concurrency) found 11 reproduced defects
before the Geny port; all fixed, each with a regression test.

### Fixed — learning (the big one)
- **Blend gate was effectively unreachable.** The decayed-McNemar scheme
  froze effective-n at ~100, so a genuinely-but-modestly-better learner
  (56–67% win-rate) could NEVER open the gate no matter how much data
  arrived — the learner was dead outside a near-oracle regime. Rebuilt as an
  **out-of-sample McNemar test**: ~25% of pairs are held out (never trained
  on), refereed with **integer** counts + a Wilson 99% lower bound. A modest
  real signal now opens the gate as data accumulates; noise (held-out
  win-rate → 50%) never does. Verified: moderate learner (0.79) → λ 0.70;
  noise (0.49) → λ 0; 17,600 random feedbacks stay shut.
- **Pairwise loss overflowed to `inf`** on large negative margins — use a
  stable softplus (`logaddexp`); gradient was already fine.

### Fixed — persistence & concurrency
- **Engine state had no lock** despite `check_same_thread=False`: concurrent
  search/index/feedback raced on the caches and ranker ("dict changed size",
  half-updated weights). One re-entrant lock now guards all public methods.
- **distill left the embedder table and vectors mutually inconsistent on
  crash.** The table lives in SQLite now (not a sidecar npz) and distill
  swaps table + all re-embedded vectors in ONE atomic transaction. Also
  removes the two-instance npz-clobber hazard.
- **Store writes had no rollback** — a mid-write exception left a partial
  transaction for the next commit. All multi-statement writes now roll back
  on failure. Reads take the lock too (single shared connection).
- **`text:<id>` params leaked forever** (never deleted on remove, duplicated
  every body). Now gated by `store_text` (default on, needed for distill)
  and cleaned up on remove().

### Fixed — Korean correctness
- Re-indexing a node with changed **links** left a dangling reverse edge
  (LINK is now stored one-directional and symmetrized at query time).
- Re-indexing with changed **tags** left stale tag-cache membership (cache
  drops on re-index).
- `remove()` didn't purge pending feedback tokens → could resurrect a deleted
  node's edges/feedback rows.
- `로` particle now strips after an ㄹ-final stem (서울로 → 서울), not just
  vowel-final.

### Changed
- distill() returns a candidate table the engine applies atomically; needs
  ≥24 teacher pairs and a held-out improvement margin (was: swap on a single
  tiny-holdout wiggle, or never swap for small corpora).
- Embedding table persists inside the db (`store_text`, `emb_path` removed).
- executor_adapter offloads index/search/distill via `asyncio.to_thread`.


## [1.1.0] — 2026-07-20

Deep-validation hardening — three real defects found by indexing **939 live
Geny prod agent notes** and stress-testing the learning loop, plus a
statistically-principled blend gate.

### Fixed
- **feedback() re-serialized the 32 MB embedding table every call** (zlib
  compress dominated at 14 s / 20 feedbacks). The embedding table is static
  between distills — feedback now persists ONLY the ~1 KB ranker. **~200×
  faster: 0.85 s → 4 ms per feedback.**
- **Blend gate could open on label noise** (numeric win-lead of ~51% over
  thousands of noisy feedbacks). Replaced the win-count heuristic with a
  **McNemar paired test on discordant pairs + a Wilson 99.9% lower bound**
  (decayed so it tracks recent skill). Random feedback now keeps λ=0; a
  genuine signal opens it immediately. Verified: 17,600 random feedbacks →
  blend 0.00; genuine preference → blend 0.70 at feedback #1.
- **Meta tags densified the graph toward a clique** (real vaults tag nearly
  every note `execution`/`success`; PPR then dominated query latency).
  Tag-edge derivation now drops tags present on >30% of the corpus — no
  topical signal, and it restores fast PPR. Real-vault natural-query latency
  ~5–14 ms at 939 notes.

### Added
- `eval/real_vault.py` (index real Obsidian-format agent notes),
  `eval/learning_sim.py` (4 ground-truth learning scenarios),
  `eval/ablation.py`. Two regression guards: blend-gate noise resistance and
  feedback-loop speed.


## [1.0.0] — 2026-07-20

Korean-first upgrade — evidence-based, measured on a committed MIRACL-ko
harness (213 human-judged queries / 2,835 passages). Hybrid nDCG@10
**0.527 → 0.593**, MRR@10 0.519 → 0.597, R@1 0.385 → 0.465 vs 0.1.0, with
no morphological analyzer and still zero API calls.

### Added
- `hangul` module: exact-arithmetic jamo decomposition emitting COMPATIBILITY
  jamo (never NFD's conjoining block), empty-종성 padding, 받침 detection,
  and a guarded 조사/어미 stripper (closed longest-match lists, ≥2-syllable
  stems, 받침-agreement gate for single-syllable particles, '의' never
  stripped alone, 어미 restricted to the 하다/되다 families).
- Two token streams (NTCIR/ACL-guided): LEXICAL for BM25 — words + guarded
  stems + overlapping syllable BIGRAMS (trigrams measured harmful: −2.5
  nDCG) + cross-space bigrams for 붙여쓰기; EMBEDDING adds padded jamo
  3/5-grams (jamo in the BM25 index measured 3× slower for no gain).
- Bloom-style k=2 hash embedding (two independent bucket views per token).
- BM25F-lite title boost; heuristic re-weighted from harness measurements.
- Offline Korean eval harness (`eval/`): committed MIRACL-ko fixture
  (Apache-2.0) + 6 synthetic robustness axes (조사/띄어쓰기/자모오타/복합명사/
  한영혼용/동음이의어) + ablation grid; CI quality-floor regression test
  (nDCG ≥ 0.55, MRR ≥ 0.55, R@5 ≥ 0.70).
- 20 Korean-specific tests (37 total).

### Measured ablations (first published isolation numbers)
- syllable bigrams +2.8 nDCG · jamo embedding stream +1.8 · guarded
  조사-strip +1.6 · trigrams −2.5 (excluded) · cross-space ≈0 on clean
  (kept as 붙여쓰기 insurance).


## [0.1.0] — 2026-07-20

Initial release — the Synapse engine.

- Hybrid retrieval: BM25 (unicode words + char 2/3-grams, Korean-friendly)
  ∪ local static hash embeddings (65,536×256 fp16), RRF-fused seeds.
- Graph layer: typed edges (LINK / TAG / KNN / CO-ACCESS) with one
  Personalized PageRank per type as ranking features; adjacency cached and
  row-normalized once per mutation.
- Learning: Hebbian co-access reinforcement with lazy decay; pairwise
  logistic online SGD ranker (14→16→1 MLP, numpy) behind a safety blend
  gate (heuristic performance floor); ε-greedy tail exploration.
- Distillation: fit the embedding table to caller-provided teacher vectors
  (Adam, geometry-correlation swap gate) — zero API calls ever.
- One SQLite file (WAL) + fp16 npz table; config via args / GMA_* env / .env.
- geny-executor duck-typed adapter (SynapseVectorHandle, SynapseRetriever).
- 17 tests; bench: 0.7 ms/doc indexing, 8.7 ms/query @ 2,000 docs.
