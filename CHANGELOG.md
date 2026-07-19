# Changelog

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
