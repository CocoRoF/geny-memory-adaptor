"""Tokenizer — unicode word tokens + character n-grams, language-agnostic.

Korean (and other unsegmented/agglutinative text) is covered WITHOUT a
morphological analyzer: character 2/3-grams give BM25 and the hash embedder
sub-word units to match on ("리듬게임" ↔ "리듬게임의" share 3 of 4 bigrams).
Latin words are kept whole (lowercased) and additionally n-grammed only when
short enough to matter.

Deterministic, dependency-free, and shared by BOTH the BM25 index and the
hash embedding layer so their views of a text never diverge.
"""

from __future__ import annotations

import re
from typing import Iterable, List

_WORD = re.compile(r"[\w']+", re.UNICODE)

#: Tokens that carry no signal (tiny stoplist — BM25's IDF handles the rest).
_STOP = {"the", "a", "an", "of", "to", "and", "is", "in", "it", "i", "you"}


def _ngrams(word: str, sizes: Iterable[int]) -> List[str]:
    out: List[str] = []
    for n in sizes:
        if len(word) <= n:
            continue
        out.extend(word[i : i + n] for i in range(len(word) - n + 1))
    return out


def tokenize(text: str, *, char_ngrams: Iterable[int] = (2, 3), limit: int = 2048) -> List[str]:
    """Text → word tokens + char n-grams (order preserved, capped at *limit*)."""
    tokens: List[str] = []
    for match in _WORD.finditer(text.lower()):
        word = match.group()
        if word in _STOP or word.isdigit() and len(word) > 6:
            continue
        tokens.append(word)
        # Sub-word units: essential for unsegmented scripts, harmless for Latin.
        if any(ord(c) > 0x2E80 for c in word) or len(word) > 3:
            tokens.extend(_ngrams(word, char_ngrams))
        if len(tokens) >= limit:
            break
    return tokens[:limit]


def fnv1a(token: str, buckets: int) -> int:
    """FNV-1a 64-bit hash → bucket index (stable across processes/platforms)."""
    h = 0xCBF29CE484222325
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % buckets
