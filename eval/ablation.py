"""Ablation grid — isolate each Korean upgrade's contribution.

Several of these (guarded josa-strip isolation, cross-space bigrams, jamo in
the embedding stream) have NO published A/B numbers — this produces them.

Run: python eval/ablation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_eval import DATA, build_engine, evaluate, load_jsonl, mean  # noqa: E402

GRID = {
    "v3-full":        {},
    "-suffix_strip":  {"suffix_strip": False},
    "-cross_space":   {"cross_space": False},
    "-jamo(embed)":   {"jamo_ngrams": ()},
    "+trigram":       {"char_ngrams": (2, 3)},
    "-title_boost":   {"title_boost": 1.0},
    "title_boost=3":  {"title_boost": 3.0},
    "-bigram(word만)": {"char_ngrams": ()},
}


def main() -> None:
    docs = load_jsonl(DATA / "corpus.jsonl")
    queries = load_jsonl(DATA / "queries_clean.jsonl")
    from collections import defaultdict
    positives = defaultdict(set)
    for line in (DATA / "qrels.tsv").open(encoding="utf-8"):
        qid, _, docid, grade = line.split("\t")
        if int(grade) > 0:
            positives[qid].add(docid)

    print(f"{'config':<16}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR':>7}{'nDCG':>7}{'idx_s':>7}{'ms/q':>7}")
    for name, overrides in GRID.items():
        t0 = time.perf_counter()
        mem = build_engine(overrides, docs)
        idx_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = evaluate(mem, queries, positives)
        msq = (time.perf_counter() - t0) / max(1, len(res)) * 1000
        print(f"{name:<16}{mean(res,'r1'):>7.3f}{mean(res,'r5'):>7.3f}"
              f"{mean(res,'r10'):>7.3f}{mean(res,'mrr'):>7.3f}"
              f"{mean(res,'ndcg'):>7.3f}{idx_s:>7.1f}{msq:>7.1f}")
        mem.close()


if __name__ == "__main__":
    main()
