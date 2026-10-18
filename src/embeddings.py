"""Generate sentence-transformer embeddings for the corpus and log timing."""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CORPUS_FILE = ROOT / "data" / "processed" / "clean_corpus.jsonl"
EMB_FILE = ROOT / "data" / "processed" / "embeddings.npy"
IDS_FILE = ROOT / "data" / "processed" / "embedding_ids.json"
LOG_FILE = ROOT / "logs" / "embeddings.log"

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64


def setup_logger() -> logging.Logger:
    """Log to both the console and logs/embeddings.log."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    logger = logging.getLogger("embeddings")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger


log = setup_logger()


def load_corpus(path: Path = CORPUS_FILE) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["id"])
            texts.append(rec["text"])
    return ids, texts


def load_model(name: str = MODEL_NAME) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    model = SentenceTransformer(name, device=device)
    log.info(f"Loaded {name} on {device} in {time.perf_counter() - start:.2f}s "
             f"(max tokens: {model.max_seq_length})")
    return model


def count_truncated(model: SentenceTransformer, texts: list[str]) -> int:
    """How many texts are longer than the model can read."""
    limit = model.max_seq_length
    return sum(1 for t in texts if len(model.tokenizer.encode(t)) > limit)


def _sync(model: SentenceTransformer) -> None:
    # GPU work runs in the background; wait for it so the timing is accurate
    if model.device.type == "cuda":
        torch.cuda.synchronize()


def embed_texts(model: SentenceTransformer, texts: list[str],
                batch_size: int = DEFAULT_BATCH_SIZE,
                show_progress: bool = True) -> tuple[np.ndarray, dict]:
    """Embed texts and return (embeddings, timing stats)."""
    if not texts:
        raise ValueError("No texts to embed.")

    model.encode(texts[:2], batch_size=2)  # warm-up, not timed
    _sync(model)

    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    _sync(model)
    elapsed = time.perf_counter() - start

    stats = {
        "n_texts": len(texts),
        "batch_size": batch_size,
        "seconds": round(elapsed, 2),
        "texts_per_sec": round(len(texts) / elapsed, 1),
        "ms_per_text": round(1000 * elapsed / len(texts), 2),
        "dim": embeddings.shape[1],
    }
    return embeddings, stats


def benchmark(model: SentenceTransformer, texts: list[str],
              batch_sizes=(16, 32, 64, 128), sample: int = 500) -> int:
    """Try several batch sizes on a sample and return the fastest."""
    sample_texts = texts[:sample]
    results = []
    for bs in batch_sizes:
        _, s = embed_texts(model, sample_texts, bs, show_progress=False)
        log.info(f"Benchmark batch_size={bs}: {s['texts_per_sec']} texts/s, "
                 f"{s['ms_per_text']} ms/text")
        results.append(s)
    best = max(results, key=lambda s: s["texts_per_sec"])["batch_size"]
    log.info(f"Fastest batch size: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Embed the clean corpus.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--benchmark", action="store_true",
                        help="test several batch sizes first and use the fastest")
    args = parser.parse_args()

    ids, texts = load_corpus()
    log.info(f"Loaded {len(texts)} documents from {CORPUS_FILE.name}")

    model = load_model()

    n_trunc = count_truncated(model, texts)
    log.info(f"{n_trunc} of {len(texts)} texts ({100 * n_trunc / len(texts):.1f}%) "
             f"exceed {model.max_seq_length} tokens and will be truncated")

    batch_size = benchmark(model, texts) if args.benchmark else args.batch_size

    embeddings, stats = embed_texts(model, texts, batch_size)
    log.info(f"Embedding done: {stats}")

    assert len(ids) == len(embeddings), "ID count and embedding count differ"
    np.save(EMB_FILE, embeddings)
    IDS_FILE.write_text(json.dumps(ids), encoding="utf-8")
    log.info(f"Saved {embeddings.shape} to {EMB_FILE.name} and IDs to {IDS_FILE.name}")


if __name__ == "__main__":
    main()