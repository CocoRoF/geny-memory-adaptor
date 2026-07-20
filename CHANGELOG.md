# Changelog

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
