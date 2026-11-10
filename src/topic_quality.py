"""Check topic quality: automatic metrics, a sheet for manual review, and its summary."""
import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOC_TOPICS_FILE = PROCESSED / "doc_topics.jsonl"
TOPIC_INFO_FILE = PROCESSED / "topic_info.csv"
EMB_FILE = PROCESSED / "embeddings.npy"
IDS_FILE = PROCESSED / "embedding_ids.json"
TOPIC_NAMES_FILE = PROCESSED / "topic_names.json"
REVIEW_FILE = ROOT / "eval" / "topic_review.csv"   # committed: it's manual work
METRICS_FILE = ROOT / "logs" / "topic_metrics.json"
SUMMARY_FILE = ROOT / "logs" / "topic_review_summary.json"

N_SAMPLES = 5
VERDICTS = ("yes", "partly", "no")


def load():
    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    emb = np.load(EMB_FILE)
    with open(DOC_TOPICS_FILE, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    if [d["id"] for d in docs] != ids:
        sys.exit("doc_topics.jsonl does not match the embeddings. Re-run src/topics.py.")
    info = pd.read_csv(TOPIC_INFO_FILE)
    return docs, emb, info


def top_words(info: pd.DataFrame) -> dict[int, list[str]]:
    out = {}
    for _, row in info.iterrows():
        try:
            words = ast.literal_eval(row["Representation"])
        except (ValueError, SyntaxError):
            words = []
        out[int(row["Topic"])] = words
    return out


def cohesion_and_samples(docs: list[dict], emb: np.ndarray) -> dict:
    """Per topic: mean similarity of its papers to the topic center, and the closest papers."""
    topics = np.array([d["topic"] for d in docs])
    result = {}
    for t in sorted(set(topics.tolist())):
        if t == -1:
            continue
        idx = np.where(topics == t)[0]
        vecs = emb[idx]
        centroid = vecs.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        sims = vecs @ centroid  # embeddings are normalized, so this is cosine similarity
        result[t] = {
            "cohesion": float(sims.mean()),
            "samples": idx[np.argsort(-sims)][:N_SAMPLES].tolist(),
        }
    return result


# ---------- commands ----------

def cmd_metrics() -> None:
    docs, emb, info = load()
    words = top_words(info)
    topic_ids = [t for t in words if t != -1]
    n = len(docs)

    all_top = [w for t in topic_ids for w in words[t][:10]]
    diversity = len(set(all_top)) / len(all_top) if all_top else 0.0

    coh = cohesion_and_samples(docs, emb)
    global_center = emb.mean(axis=0)
    global_center /= np.linalg.norm(global_center)
    corpus_baseline = float((emb @ global_center).mean())

    sizes = Counter(d["topic"] for d in docs)
    largest_t, largest_n = max(((t, c) for t, c in sizes.items() if t != -1),
                               key=lambda x: x[1])

    metrics = {
        "documents": n,
        "topics": len(topic_ids),
        "outliers": sizes.get(-1, 0),
        "outlier_pct": round(100 * sizes.get(-1, 0) / n, 1),
        "reassigned": sum(1 for d in docs if d.get("reassigned")),
        "short_docs": sum(1 for d in docs if d.get("short_doc")),
        "topic_diversity": round(diversity, 3),
        "mean_cohesion": round(float(np.mean([c["cohesion"] for c in coh.values()])), 3),
        "corpus_baseline_cohesion": round(corpus_baseline, 3),
        "largest_topic": {"topic": largest_t, "papers": largest_n,
                          "pct": round(100 * largest_n / n, 1)},
        "weakest_topics": [
            {"topic": t, "cohesion": round(c["cohesion"], 3)}
            for t, c in sorted(coh.items(), key=lambda x: x[1]["cohesion"])[:5]
        ],
    }

    print(f"Topics: {metrics['topics']} | outliers: {metrics['outliers']} "
          f"({metrics['outlier_pct']}%) | reassigned: {metrics['reassigned']} "
          f"| short docs: {metrics['short_docs']}")
    print(f"Topic diversity: {metrics['topic_diversity']} (1.0 = no shared top words)")
    print(f"Mean cohesion: {metrics['mean_cohesion']} "
          f"(whole corpus baseline: {metrics['corpus_baseline_cohesion']})")
    print(f"Largest topic: {largest_t} with {largest_n} papers "
          f"({metrics['largest_topic']['pct']}%)")
    print("Weakest topics:", ", ".join(
        f"{w['topic']} ({w['cohesion']})" for w in metrics["weakest_topics"]))

    METRICS_FILE.parent.mkdir(exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved to {METRICS_FILE}")


def cmd_review(force: bool) -> None:
    """Write a sheet with one row per topic for manual checking."""
    if REVIEW_FILE.exists() and not force:
        sys.exit(f"{REVIEW_FILE} already exists. Use --force to overwrite "
                 f"(this deletes your review).")

    docs, emb, info = load()
    words = top_words(info)
    coh = cohesion_and_samples(docs, emb)
    sizes = Counter(d["topic"] for d in docs)
    labels = {d["topic"]: d["topic_label"] for d in docs}

    REVIEW_FILE.parent.mkdir(exist_ok=True)
    with open(REVIEW_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "papers", "label", "top_words", "cohesion",
                         "sample_titles", "coherent", "better_name", "notes"])
        for t in sorted(coh, key=lambda t: -sizes[t]):
            titles = " || ".join(docs[i]["title"] for i in coh[t]["samples"])
            writer.writerow([t, sizes[t], labels[t], ", ".join(words.get(t, [])[:10]),
                             round(coh[t]["cohesion"], 3), titles, "", "", ""])
    print(f"Wrote {len(coh)} topics to {REVIEW_FILE}. "
          f"Fill in coherent (yes/partly/no), better_name and notes.")


def cmd_summarize() -> None:
    if not REVIEW_FILE.exists():
        sys.exit(f"{REVIEW_FILE} not found. Run: python src/topic_quality.py review")

    with open(REVIEW_FILE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    reviewed = [r for r in rows if r["coherent"].strip().lower() in VERDICTS]
    if not reviewed:
        sys.exit("No reviewed topics yet. Fill in the coherent column first.")

    topic_counts, paper_counts = Counter(), Counter()
    cohesion_by_verdict = defaultdict(list)
    for r in reviewed:
        v = r["coherent"].strip().lower()
        topic_counts[v] += 1
        paper_counts[v] += int(r["papers"])
        cohesion_by_verdict[v].append(float(r["cohesion"]))
    total_topics, total_papers = len(reviewed), sum(paper_counts.values())

    names = {int(r["topic"]): r["better_name"].strip() for r in rows if r["better_name"].strip()}
    TOPIC_NAMES_FILE.write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "reviewed_topics": total_topics,
        "by_topic": {v: round(100 * topic_counts[v] / total_topics, 1) for v in VERDICTS},
        "by_paper": {v: round(100 * paper_counts[v] / total_papers, 1) for v in VERDICTS},
        "mean_cohesion_by_verdict": {v: round(float(np.mean(c)), 3)
                                     for v, c in cohesion_by_verdict.items()},
        "not_coherent": [{"topic": int(r["topic"]), "label": r["label"], "notes": r["notes"]}
                         for r in reviewed if r["coherent"].strip().lower() == "no"],
        "renamed_topics": len(names),
    }

    print(f"Reviewed {total_topics} topics")
    for v in VERDICTS:
        print(f"  {v:7} {summary['by_topic'][v]:5.1f}% of topics | "
              f"{summary['by_paper'][v]:5.1f}% of papers")
    print("Mean cohesion by verdict:", summary["mean_cohesion_by_verdict"])
    print(f"Saved {len(names)} better names to {TOPIC_NAMES_FILE.name}")

    SUMMARY_FILE.parent.mkdir(exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {SUMMARY_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Topic quality checks.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("metrics", help="automatic quality numbers")
    p_review = sub.add_parser("review", help="create the sheet for manual review")
    p_review.add_argument("--force", action="store_true")
    sub.add_parser("summarize", help="summarize your manual review")
    args = parser.parse_args()

    if args.command == "metrics":
        cmd_metrics()
    elif args.command == "review":
        cmd_review(args.force)
    elif args.command == "summarize":
        cmd_summarize()


if __name__ == "__main__":
    main()