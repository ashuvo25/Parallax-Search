"""Discover corpus themes with BERTopic, reusing the saved embeddings.

Edge cases handled:
- outliers (topic -1) are moved to their closest topic when similar enough
- short documents are flagged, since their topics are less reliable
- words that appear in many topics (jargon) are reported as stop-word candidates
"""
import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import MODEL_NAME, load_model  # noqa: E402

try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN_PKG = True
except ImportError:  # hdbscan can fail to install on Windows
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN_PKG = False

PROCESSED = ROOT / "data" / "processed"
CORPUS_FILE = PROCESSED / "clean_corpus.jsonl"
EMB_FILE = PROCESSED / "embeddings.npy"
IDS_FILE = PROCESSED / "embedding_ids.json"
DOC_TOPICS_FILE = PROCESSED / "doc_topics.jsonl"
TOPIC_INFO_FILE = PROCESSED / "topic_info.csv"
MODEL_DIR = ROOT / "models" / "bertopic"
REPORT_DIR = ROOT / "reports"
LOG_FILE = ROOT / "logs" / "topics.log"

SHORT_DOC_WORDS = 60              # papers shorter than this get less reliable topics
DEFAULT_OUTLIER_THRESHOLD = 0.5   # min similarity to move an outlier into a topic
GENERIC_WORD_SHARE = 0.25         # a word in 25%+ of topics is too generic

# Words found in almost every abstract. They say nothing about the topic.
DOMAIN_STOP_WORDS = {
    "paper", "propose", "proposed", "proposes", "method", "methods", "approach",
    "approaches", "results", "result", "show", "shows", "based", "using", "use",
    "used", "framework", "performance", "new", "novel", "existing", "work",
    "study", "task", "tasks", "model", "models", "data", "state", "art",
    "demonstrate", "experiments", "experimental", "present", "introduce",
    "achieve", "achieves", "significantly", "improve", "improves", "improved",
}


def setup_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("topics")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger


log = setup_logger()


def load_data():
    """Return ids, texts, titles, word counts and embeddings, all in the same order."""
    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    embeddings = np.load(EMB_FILE)

    records = {}
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[rec["id"]] = rec

    missing = [i for i in ids if i not in records]
    if missing:
        raise KeyError(f"{len(missing)} embedding IDs are not in the corpus. "
                       f"Re-run src/embeddings.py.")

    docs = [records[i]["text"] for i in ids]
    titles = [records[i]["title"] for i in ids]
    n_words = [records[i]["n_words"] for i in ids]
    return ids, docs, titles, n_words, embeddings


def build_model(embedding_model, min_cluster_size: int):
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,  # same topics on every run
    )

    hdbscan_kwargs = {
        "min_cluster_size": min_cluster_size,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
    }
    if HAS_HDBSCAN_PKG:
        hdbscan_kwargs["prediction_data"] = True
    hdbscan_model = HDBSCAN(**hdbscan_kwargs)

    vectorizer = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=5,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=10,
        calculate_probabilities=False,
        verbose=True,
    )
    return topic_model, vectorizer


def find_generic_words(topic_model, share: float = GENERIC_WORD_SHARE) -> list[tuple[str, int]]:
    """Words that appear in the top 10 of many topics: likely jargon, not a theme."""
    topic_ids = [t for t in topic_model.get_topics() if t != -1]
    counter = Counter()
    for t in topic_ids:
        for word, _ in topic_model.get_topic(t)[:10]:
            counter[word] += 1
    limit = max(2, int(share * len(topic_ids)))
    return [(w, c) for w, c in counter.most_common() if c >= limit]


def main():
    parser = argparse.ArgumentParser(description="Topic modeling with BERTopic.")
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--nr-topics", type=int, default=None,
                        help="merge similar topics down to this number")
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD,
                        help="min similarity to move an outlier into a topic (0 = keep all outliers)")
    args = parser.parse_args()

    if not HAS_HDBSCAN_PKG:
        log.warning("hdbscan package not found, using scikit-learn's HDBSCAN.")

    ids, docs, titles, n_words, embeddings = load_data()
    log.info(f"Loaded {len(docs)} documents")

    topic_model, vectorizer = build_model(load_model(), args.min_cluster_size)

    start = time.perf_counter()
    topics, _ = topic_model.fit_transform(docs, embeddings)
    if args.nr_topics:
        topic_model.reduce_topics(docs, nr_topics=args.nr_topics)
        topics = list(topic_model.topics_)
    log.info(f"Fitted in {time.perf_counter() - start:.1f}s")

    probs = topic_model.probabilities_
    original_topics = list(topics)
    outliers_before = sum(1 for t in topics if t == -1)

    # --- edge case: short documents ---
    short = [n < SHORT_DOC_WORDS for n in n_words]
    n_short = sum(short)
    overall_rate = 100 * outliers_before / len(docs)
    if n_short:
        short_outliers = sum(1 for s, t in zip(short, original_topics) if s and t == -1)
        log.info(f"Short documents (<{SHORT_DOC_WORDS} words): {n_short} | "
                 f"outlier rate {100 * short_outliers / n_short:.1f}% "
                 f"vs {overall_rate:.1f}% overall")
    else:
        log.info(f"No documents shorter than {SHORT_DOC_WORDS} words")

    # --- edge case: outliers ---
    if args.outlier_threshold > 0 and outliers_before:
        new_topics = topic_model.reduce_outliers(
            docs, topics, strategy="embeddings",
            embeddings=embeddings, threshold=args.outlier_threshold,
        )
        topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer)
        topics = list(new_topics)
    reassigned = [o == -1 and t != -1 for o, t in zip(original_topics, topics)]
    outliers_after = sum(1 for t in topics if t == -1)
    log.info(f"Outliers: {outliers_before} ({overall_rate:.1f}%) -> {outliers_after} "
             f"({100 * outliers_after / len(docs):.1f}%), {sum(reassigned)} reassigned "
             f"(threshold {args.outlier_threshold})")

    # --- edge case: jargon ---
    generic = find_generic_words(topic_model)
    if generic:
        log.info("Words in many topics (consider adding to DOMAIN_STOP_WORDS): "
                 + ", ".join(f"{w} ({c})" for w, c in generic))

    # labels
    labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, separator=", ")
    topic_model.set_topic_labels(labels)
    info = topic_model.get_topic_info()
    label_map = dict(zip(info["Topic"], info["CustomName"]))
    label_map[-1] = "Unassigned"

    n_topics = int((info["Topic"] != -1).sum())
    log.info(f"Found {n_topics} topics")
    for _, row in info[info["Topic"] != -1].head(10).iterrows():
        log.info(f"  Topic {row['Topic']:>3} | {row['Count']:>4} papers | {row['CustomName']}")

    # per-document topics
    with open(DOC_TOPICS_FILE, "w", encoding="utf-8") as f:
        for i, (doc_id, t) in enumerate(zip(ids, topics)):
            t = int(t)
            prob = None
            if probs is not None and not reassigned[i]:
                prob = round(float(probs[i]), 4)
            f.write(json.dumps({
                "id": doc_id,
                "title": titles[i],
                "topic": t,
                "topic_label": label_map[t],
                "topic_prob": prob,
                "reassigned": bool(reassigned[i]),
                "short_doc": bool(short[i]),
            }, ensure_ascii=False) + "\n")

    info[["Topic", "Count", "CustomName", "Representation"]].to_csv(
        TOPIC_INFO_FILE, index=False, encoding="utf-8")

    MODEL_DIR.parent.mkdir(exist_ok=True)
    topic_model.save(
        str(MODEL_DIR),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=f"sentence-transformers/{MODEL_NAME}",
    )

    REPORT_DIR.mkdir(exist_ok=True)
    try:
        topic_model.visualize_barchart(top_n_topics=20, n_words=8, custom_labels=True) \
            .write_html(REPORT_DIR / "topics_barchart.html")
        topic_model.visualize_topics(custom_labels=True) \
            .write_html(REPORT_DIR / "topics_map.html")
    except Exception as e:
        log.warning(f"Could not create charts: {e}")

    log.info(f"Saved {DOC_TOPICS_FILE.name}, {TOPIC_INFO_FILE.name}, model and charts")


if __name__ == "__main__":
    main()