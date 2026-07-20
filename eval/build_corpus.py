"""One-time builder for the committed Korean eval fixture (MIRACL-ko dev).

Downloads the RAW MIRACL files (TREC topics/qrels + jsonl.gz corpus parts) via
huggingface_hub — no `datasets` library, no dataset scripts. Emits into
eval/data/ (all Apache-2.0, same license as the source):

    corpus.jsonl         {docid, title, text}   (~1,500 passages)
    queries_clean.jsonl  {qid, text}            (~200 queries)
    qrels.tsv            qid \t 0 \t docid \t grade

Corpus = every judged doc (positives + judged hard negatives) for the kept
queries, padded with random unrelated-article passages. Unjudged siblings of
positive articles are EXCLUDED from the pad pool (relevant-but-unjudged docs
would poison Recall).

Run:  python eval/build_corpus.py     (needs `pip install huggingface_hub`)
"""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

PAD_TO = 1500
SEED = 41
OUT = Path(__file__).parent / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    topics_path = hf_hub_download(
        "miracl/miracl", "miracl-v1.0-ko/topics/topics.miracl-v1.0-ko-dev.tsv",
        repo_type="dataset")
    qrels_path = hf_hub_download(
        "miracl/miracl", "miracl-v1.0-ko/qrels/qrels.miracl-v1.0-ko-dev.tsv",
        repo_type="dataset")

    topics: dict[str, str] = {}
    for line in open(topics_path, encoding="utf-8"):
        qid, _, text = line.rstrip("\n").partition("\t")
        if qid and text:
            topics[qid] = text

    qrels: list[tuple[str, str, int]] = []
    by_query: dict[str, list[int]] = defaultdict(list)
    judged_ids: set[str] = set()
    for line in open(qrels_path, encoding="utf-8"):
        parts = line.split()
        if len(parts) != 4:
            continue
        qid, _, docid, grade = parts[0], parts[1], parts[2], int(parts[3])
        qrels.append((qid, docid, grade))
        by_query[qid].append(grade)
        judged_ids.add(docid)

    # Keep queries that have ≥1 positive judgment and a topic text.
    keep = {qid for qid, grades in by_query.items() if any(g > 0 for g in grades) and qid in topics}
    qrels = [(q, d, g) for q, d, g in qrels if q in keep]
    judged_ids = {d for _, d, _ in qrels}
    pos_articles = {d.split("#", 1)[0] for q, d, g in qrels if g > 0}

    # Scan the 3 corpus parts once: collect judged docs + reservoir-pad.
    judged_docs: dict[str, dict] = {}
    pad_needed_estimate = max(0, PAD_TO - len(judged_ids))
    pad: list[dict] = []
    seen = 0
    for part in range(3):
        path = hf_hub_download(
            "miracl/miracl-corpus", f"miracl-corpus-v1.0-ko/docs-{part}.jsonl.gz",
            repo_type="dataset")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for raw in f:
                row = json.loads(raw)
                docid = row["docid"]
                if docid in judged_ids:
                    judged_docs[docid] = {"docid": docid, "title": row.get("title") or "",
                                          "text": row.get("text") or ""}
                    continue
                if docid.split("#", 1)[0] in pos_articles:
                    continue  # unjudged sibling of a positive — exclude
                seen += 1
                item = {"docid": docid, "title": row.get("title") or "",
                        "text": row.get("text") or ""}
                if len(pad) < pad_needed_estimate:
                    pad.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < pad_needed_estimate:
                        pad[j] = item

    # Drop qrels whose doc never surfaced (shouldn't happen, but be exact).
    qrels = [(q, d, g) for q, d, g in qrels if d in judged_docs]
    docs = list(judged_docs.values()) + pad[: max(0, PAD_TO - len(judged_docs))]

    with (OUT / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with (OUT / "queries_clean.jsonl").open("w", encoding="utf-8") as f:
        for qid in sorted(keep):
            f.write(json.dumps({"qid": qid, "text": topics[qid]}, ensure_ascii=False) + "\n")
    with (OUT / "qrels.tsv").open("w", encoding="utf-8") as f:
        for qid, docid, grade in qrels:
            f.write(f"{qid}\t0\t{docid}\t{grade}\n")

    print(f"queries={len(keep)} judged_docs={len(judged_docs)} "
          f"pad={len(docs) - len(judged_docs)} corpus={len(docs)} qrels={len(qrels)}")


if __name__ == "__main__":
    main()
