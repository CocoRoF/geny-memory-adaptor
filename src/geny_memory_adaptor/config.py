"""Configuration — explicit args first, environment second, defaults last.

Every knob can arrive three ways (highest precedence first):

1. Constructor arguments: ``SynapseMemory(path=..., dim=...)``
2. Environment variables with the ``GMA_`` prefix (a ``.env`` file can be
   loaded with :func:`load_dotenv` — a tiny dependency-free parser)
3. The dataclass defaults below

The engine itself never talks to the network; the only "credential"-shaped
value is the optional teacher-embedding source used for distillation, and even
that is data the caller hands in (vectors), not a key this library uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

_ENV_PREFIX = "GMA_"


def load_dotenv(path: str | os.PathLike = ".env", *, override: bool = False) -> Dict[str, str]:
    """Minimal ``.env`` loader (KEY=VALUE lines, ``#`` comments, quotes).

    Returns the parsed map and sets ``os.environ`` (existing values win unless
    ``override``). No dependency on python-dotenv.
    """
    parsed: Dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return parsed
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


@dataclass
class SynapseConfig:
    """All engine knobs. See module docstring for the precedence rules."""

    # ── storage ──
    #: SQLite database path. ``":memory:"`` for ephemeral (tests).
    path: str = "synapse.db"
    #: Embedding table sidecar (npz). Default: ``<path>.emb.npz`` next to the db.
    emb_path: str = ""

    # ── embedding layer ──
    #: Hash-bucket vocabulary size (power of two).
    vocab_size: int = 65_536
    #: Local embedding dimension.
    dim: int = 256
    #: Deterministic seed for the initial hash-projection table.
    seed: int = 41

    # ── tokenizer ──
    #: Character n-gram sizes indexed alongside word tokens (Korean-friendly).
    char_ngrams: tuple = (2, 3)
    #: Per-document token cap (indexing) / per-query cap.
    max_doc_tokens: int = 2048
    max_query_tokens: int = 256

    # ── retrieval ──
    #: Seed set sizes for the two retrievers before fusion.
    bm25_seed_k: int = 24
    vector_seed_k: int = 24
    #: BM25 parameters.
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    #: PPR restart probability and iteration cap (per edge type).
    ppr_alpha: float = 0.5
    ppr_iters: int = 20
    #: Graph expansion frontier cap (nodes added beyond seeds).
    graph_expand_k: int = 16
    #: Final results returned by default.
    top_k: int = 8

    # ── graph edges ──
    #: Semantic k-NN edges derived at index time.
    knn_edges: int = 6
    knn_min_sim: float = 0.25
    #: Tag-edge fanout cap per tag (anti-hub).
    tag_fanout: int = 6
    #: Hebbian co-access learning rate / weekly decay / prune floor.
    hebb_eta: float = 0.3
    hebb_decay: float = 0.9
    hebb_prune: float = 0.05

    # ── ranker / learning ──
    #: Hidden width of the tiny MLP.
    hidden: int = 16
    #: Online SGD learning rate + L2.
    lr: float = 0.05
    l2: float = 1e-4
    #: Replay buffer size (feedback rows kept for mini-batch refreshes).
    replay_cap: int = 4096
    #: Blend gate: the learned score joins the heuristic only after this many
    #: labelled events AND a better online AUC (safety floor).
    blend_min_events: int = 100
    #: Exploration: probability of swapping one tail slot with a random candidate.
    epsilon: float = 0.05

    # ── distillation ──
    #: Distillation mini-batch epochs / learning rate (Adam).
    distill_epochs: int = 4
    distill_lr: float = 1e-3
    distill_batch: int = 64

    #: Free-form extras for forward compatibility.
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, *, dotenv: Optional[str] = None, **overrides: Any) -> "SynapseConfig":
        """Build a config from ``GMA_*`` environment variables + overrides.

        ``GMA_PATH``, ``GMA_DIM``, ``GMA_VOCAB_SIZE``, … (upper-cased field
        names). Explicit ``overrides`` win over env; env wins over defaults.
        """
        if dotenv:
            load_dotenv(dotenv)
        values: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "extras":
                continue
            raw = os.environ.get(_ENV_PREFIX + f.name.upper())
            if raw is None:
                continue
            values[f.name] = _coerce(raw, f.type)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def _coerce(raw: str, annotation: Any) -> Any:
    text = str(annotation)
    if "bool" in text:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if "int" in text:
        return int(raw)
    if "float" in text:
        return float(raw)
    if "tuple" in text:
        return tuple(int(x) for x in raw.replace(",", " ").split())
    return raw
