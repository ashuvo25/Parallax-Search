"""Test retrieval latency and quality against the ChromaDB collection."""
import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import MODEL_NAME, load_model  # noqa: E402
from src.vector_store import CORPUS_FILE, get_client, get_collection  # noqa: E402

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "retrieval.log"
REPORT_FILE = LOG_DIR / "retrieval_report.json"

TEST_QUERIES = {
    "short": [
        "transformers",
        "reinforcement learning",
        "diffusion models",
    ],
    "medium": [
        "making large language models safer",
        "graph neural networks for molecules",
        "speech recognition for low-resource languages",
    ],
    "long": [
        "how can retrieval augmented generation reduce hallucinations in question answering systems",
        "methods for evaluating the robustness of vision models against adversarial attacks",
        "using reinforcement learning from human feedback to align chatbots with user preferences",
    ],
    "non_english": [
        "réseaux de neurones pour la traduction automatique",
    ],
    "edge": [
        "a",                  # single character
        "????",               # no real words
        "the of and to in",   # only stopwords
    ],
}


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("retrieval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger


log = setup_logger()


# ---------- helpers ----------

def timed_search(model, collection, query: str, k: int):
    """Run one search. Returns (result, encode_ms, db_ms)."""
    k = max(1, min(k, collection.count()))

    t0 = time.perf_counter()
    emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    t1 = time.perf_counter()
    res = collection.query(
        query_embeddings=emb.tolist(),
        n_results=k,
        include=["metadatas", "distances"],
    )
    t2 = time.perf_counter()

    return res, (t1 - t0) * 1000, (t2 - t1) * 1000


def summarize(times_ms: list[float]) -> dict:
    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


# ---------- tests ----------

def latency_test(model, collection, k: int, repeats: int) -> tuple[list[dict], list[float]]:
    log.info(f"--- Latency by query (k={k}, {repeats} runs each) ---")
    rows, all_totals = [], []

    for category, queries in TEST_QUERIES.items():
        for query in queries:
            enc_times, db_times = [], []
            for _ in range(repeats):
                res, enc_ms, db_ms = timed_search(model, collection, query, k)
                enc_times.append(enc_ms)
                db_times.append(db_ms)
            totals = [e + d for e, d in zip(enc_times, db_times)]
            all_totals.extend(totals)

            has_result = bool(res["ids"][0])
            top_title = res["metadatas"][0][0].get("title", "") if has_result else ""
            top_score = round(1 - res["distances"][0][0] / 2, 4) if has_result else None

            row = {
                "category": category,
                "query": query,
                "encode_p50_ms": round(float(np.median(enc_times)), 2),
                "db_p50_ms": round(float(np.median(db_times)), 2),
                "total": summarize(totals),
                "top_title": top_title,
                "top_score": top_score,
            }
            rows.append(row)
            log.info(
                f"[{category:11}] {row['total']['p50_ms']:7.2f} ms "
                f"(encode {row['encode_p50_ms']:6.2f} + db {row['db_p50_ms']:6.2f}) "
                f"| {query[:45]!r} -> [{top_score}] {top_title[:50]}"
            )

    return rows, all_totals


def k_test(model, collection, repeats: int, ks=(1, 5, 10, 50, 100)) -> list[dict]:
    query = "making large language models safer"
    log.info(f"--- Latency vs k (query: {query!r}) ---")
    rows = []
    for k in ks:
        totals = []
        for _ in range(repeats):
            _, enc_ms, db_ms = timed_search(model, collection, query, k)
            totals.append(enc_ms + db_ms)
        stats = summarize(totals)
        rows.append({"k": k, **stats})
        log.info(f"k={k:<4} p50 {stats['p50_ms']:7.2f} ms | p95 {stats['p95_ms']:7.2f} ms")
    return rows


def quality_test(model, collection, n_samples: int, k: int) -> dict:
    """Search with each sampled paper's title and check if the paper comes back."""
    with open(CORPUS_FILE, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    random.seed(42)  # same sample every run, so results are comparable
    sample = random.sample(records, min(n_samples, len(records)))

    hits_at_1 = hits_at_k = 0
    reciprocal_ranks = []
    for rec in sample:
        res, _, _ = timed_search(model, collection, rec["title"], k)
        ids = res["ids"][0]
        if rec["id"] in ids:
            rank = ids.index(rec["id"]) + 1
            reciprocal_ranks.append(1 / rank)
            hits_at_k += 1
            hits_at_1 += rank == 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(sample)
    result = {
        "n_queries": n,
        "recall@1": round(hits_at_1 / n, 4),
        f"recall@{k}": round(hits_at_k / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
    }
    log.info(f"--- Quality (title -> paper, {n} samples) ---")
    log.info(f"Recall@1 {result['recall@1']:.3f} | Recall@{k} {result[f'recall@{k}']:.3f} "
             f"| MRR {result['mrr']:.3f}")
    return result


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Test retrieval latency and quality.")
    parser.add_argument("-k", type=int, default=5, help="results per query")
    parser.add_argument("--repeats", type=int, default=5, help="runs per query for timing")
    parser.add_argument("--samples", type=int, default=200, help="papers for the quality test")
    args = parser.parse_args()

    collection = get_collection(get_client())
    count = collection.count()
    if count == 0:
        log.error("Collection is empty. Run: python src/vector_store.py ingest")
        sys.exit(1)
    log.info(f"Collection has {count} records")

    model = load_model()

    for _ in range(3):  # warm-up, not timed
        timed_search(model, collection, "warm up query", args.k)

    latency_rows, all_totals = latency_test(model, collection, args.k, args.repeats)
    overall = summarize(all_totals)
    log.info(f"Overall latency: p50 {overall['p50_ms']} ms | p95 {overall['p95_ms']} ms "
             f"| max {overall['max_ms']} ms")

    k_rows = k_test(model, collection, args.repeats)
    quality = quality_test(model, collection, args.samples, args.k)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "device": str(model.device),
        "collection_size": count,
        "k": args.k,
        "repeats": args.repeats,
        "overall_latency": overall,
        "latency_by_query": latency_rows,
        "latency_by_k": k_rows,
        "quality": quality,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    main() 