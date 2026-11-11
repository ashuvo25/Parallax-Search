"""Retrieval evaluation: build a judged test set, then compute Precision@K and Recall@K.

    pool      -> candidates from semantic + keyword search, written to a CSV to judge by hand
    evaluate  -> Precision@K, Recall@K and MRR for semantic search vs a keyword baseline
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import load_model  # noqa: E402
from src.vector_store import CORPUS_FILE, get_client, get_collection, search  # noqa: E402

QUERIES_FILE = ROOT / "eval" / "retrieval_queries.csv"
JUDGMENTS_FILE = ROOT / "eval" / "retrieval_judgments.csv"
REPORT_FILE = ROOT / "logs" / "retrieval_eval.json"
PER_QUERY_FILE = ROOT / "logs" / "retrieval_eval_per_query.csv"

POOL_DEPTH = 10
DEFAULT_KS = (1, 3, 5, 10)
JUDGMENT_FIELDS = ["query_id", "query", "doc_id", "title", "abstract", "relevant"]


# ---------- data ----------

def load_queries() -> list[tuple[str, str]]:
    if not QUERIES_FILE.exists():
        sys.exit(f"{QUERIES_FILE} not found. Create it with columns: query_id,query")
    with open(QUERIES_FILE, encoding="utf-8-sig", newline="") as f:
        rows = [(r["query_id"].strip(), r["query"].strip())
                for r in csv.DictReader(f) if r.get("query", "").strip()]
    ids = [qid for qid, _ in rows]
    if len(ids) != len(set(ids)):
        sys.exit("Duplicate query_id values in the queries file.")
    return rows


def load_corpus() -> list[dict]:
    with open(CORPUS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_judgment_rows() -> list[dict]:
    if not JUDGMENTS_FILE.exists():
        return []
    with open(JUDGMENTS_FILE, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_judgments() -> dict[str, dict[str, int]]:
    """query_id -> {doc_id: 0 or 1}. Rows without a 0/1 are not judged yet."""
    judgments = defaultdict(dict)
    for row in read_judgment_rows():
        value = (row.get("relevant") or "").strip()
        if value in {"0", "1"}:
            judgments[row["query_id"]][row["doc_id"]] = int(value)
    return judgments


# ---------- search systems ----------

class DenseSearch:
    """Our semantic search (ChromaDB + all-MiniLM-L6-v2)."""

    def __init__(self):
        self.collection = get_collection(get_client())
        self.model = load_model()

    def search(self, query: str, k: int) -> list[str]:
        return [r["doc_id"] for r in search(self.collection, self.model, query, k)]


class KeywordSearch:
    """TF-IDF keyword baseline. Widens the judging pool and gives a comparison."""

    def __init__(self, records: list[dict]):
        self.ids = [r["id"] for r in records]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          min_df=2, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([r["text"] for r in records])

    def search(self, query: str, k: int) -> list[str]:
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        top = np.argsort(-scores)[:k]
        return [self.ids[i] for i in top if scores[i] > 0]


# ---------- commands ----------

def cmd_pool(depth: int) -> None:
    """Write the candidates for each query to a CSV for judging. Keeps existing judgments."""
    queries = load_queries()
    records = load_corpus()
    by_id = {r["id"]: r for r in records}
    dense, keyword = DenseSearch(), KeywordSearch(records)

    old_rows = read_judgment_rows()
    old = {(r["query_id"], r["doc_id"]): r for r in old_rows}

    rows, seen = [], set()
    rng = random.Random(42)
    for qid, query in queries:
        pool = list(dict.fromkeys(dense.search(query, depth) + keyword.search(query, depth)))
        rng.shuffle(pool)  # blind judging: no rank or source shown
        for doc_id in pool:
            rec = by_id[doc_id]
            seen.add((qid, doc_id))
            rows.append({
                "query_id": qid,
                "query": query,
                "doc_id": doc_id,
                "title": rec["title"],
                "abstract": rec["abstract"][:400],
                "relevant": old.get((qid, doc_id), {}).get("relevant", ""),
            })

    # never throw away manual work, even if a document left the pool
    kept = [r for key, r in old.items() if key not in seen and r.get("relevant", "").strip()]
    rows += [{f: r.get(f, "") for f in JUDGMENT_FIELDS} for r in kept]

    JUDGMENTS_FILE.parent.mkdir(exist_ok=True)
    with open(JUDGMENTS_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JUDGMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    judged = sum(1 for r in rows if r["relevant"].strip() in {"0", "1"})
    print(f"{len(queries)} queries, {len(rows)} candidates "
          f"({len(rows) / max(len(queries), 1):.1f} per query)")
    print(f"Already judged: {judged} | to judge: {len(rows) - judged}")
    print(f"Saved to {JUDGMENTS_FILE}. Mark each row's relevant column 1 or 0.")


def evaluate_system(name: str, search_fn, queries, judgments, ks) -> tuple[dict, list[dict]]:
    max_k = max(ks)
    per_query, latencies = [], []

    for qid, query in queries:
        if qid not in judgments:
            continue
        judged = judgments[qid]
        relevant = {d for d, v in judged.items() if v == 1}

        start = time.perf_counter()
        ranked = search_fn(query, max_k)
        latencies.append((time.perf_counter() - start) * 1000)

        first_hit = next((i for i, d in enumerate(ranked, 1) if d in relevant), None)
        row = {"system": name, "query_id": qid, "query": query, "n_relevant": len(relevant),
               "rr": round(1 / first_hit, 4) if first_hit else 0.0}
        for k in ks:
            top = ranked[:k]
            hits = sum(1 for d in top if d in relevant)
            row[f"P@{k}"] = round(hits / k, 4)
            row[f"R@{k}"] = round(hits / len(relevant), 4) if relevant else None
            row[f"unjudged@{k}"] = sum(1 for d in top if d not in judged)
        per_query.append(row)

    with_relevant = [r for r in per_query if r["n_relevant"] > 0]
    summary = {
        "system": name,
        "queries": len(per_query),
        "queries_with_relevant": len(with_relevant),
        "mrr": round(float(np.mean([r["rr"] for r in with_relevant])), 4) if with_relevant else None,
        "median_latency_ms": round(float(np.median(latencies)), 1) if latencies else None,
    }
    for k in ks:
        summary[f"P@{k}"] = round(float(np.mean([r[f"P@{k}"] for r in per_query])), 4)
        summary[f"R@{k}"] = (round(float(np.mean([r[f"R@{k}"] for r in with_relevant])), 4)
                             if with_relevant else None)
        summary[f"unjudged@{k}"] = sum(r[f"unjudged@{k}"] for r in per_query)
    return summary, per_query


def cmd_evaluate(ks: tuple[int, ...]) -> None:
    queries = load_queries()
    judgments = read_judgments()
    if not judgments:
        sys.exit("No judgments yet. Run `pool`, then fill in the relevant column.")

    complete = [qid for qid, _ in queries if qid in judgments]
    print(f"Judged queries: {len(complete)} of {len(queries)}")

    records = load_corpus()
    dense, keyword = DenseSearch(), KeywordSearch(records)
    dense.search("warm up", 1)

    summaries, all_rows = [], []
    for name, system in (("semantic", dense), ("keyword_tfidf", keyword)):
        summary, rows = evaluate_system(name, system.search, queries, judgments, ks)
        summaries.append(summary)
        all_rows += rows

    # comparison table
    header = f"{'metric':<12}" + "".join(f"{s['system']:>16}" for s in summaries)
    print("\n" + header)
    print("-" * len(header))
    for metric in [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks] + ["mrr", "median_latency_ms"]:
        print(f"{metric:<12}" + "".join(f"{str(s[metric]):>16}" for s in summaries))

    unjudged = summaries[0][f"unjudged@{max(ks)}"]
    if unjudged:
        print(f"\nNote: {unjudged} semantic results in the top {max(ks)} were never judged "
              f"and count as not relevant. Run `pool` again and judge the new rows.")

    no_rel = [r["query_id"] for r in all_rows if r["system"] == "semantic" and r["n_relevant"] == 0]
    if no_rel:
        print(f"Queries with no relevant papers (left out of recall): {', '.join(no_rel)}")

    k5 = 5 if 5 in ks else max(ks)
    worst = sorted((r for r in all_rows if r["system"] == "semantic" and r["n_relevant"] > 0),
                   key=lambda r: r[f"R@{k5}"])[:5]
    print(f"\nHardest queries for semantic search (lowest R@{k5}):")
    for r in worst:
        print(f"  {r['query_id']} R@{k5}={r[f'R@{k5}']:.2f} "
              f"({r['n_relevant']} relevant) {r['query'][:60]}")

    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text(json.dumps({"ks": list(ks), "systems": summaries}, indent=2),
                           encoding="utf-8")
    with open(PER_QUERY_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {REPORT_FILE.name} and {PER_QUERY_FILE.name} in logs/")


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation with Precision@K and Recall@K.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pool = sub.add_parser("pool", help="create the CSV of candidates to judge")
    p_pool.add_argument("--depth", type=int, default=POOL_DEPTH,
                        help="results taken from each search method per query")

    p_eval = sub.add_parser("evaluate", help="compute the metrics from your judgments")
    p_eval.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))

    args = parser.parse_args()
    if args.command == "pool":
        cmd_pool(args.depth)
    else:
        cmd_evaluate(tuple(sorted(set(args.k))))


if __name__ == "__main__":
    main()