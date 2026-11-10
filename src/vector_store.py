"""ChromaDB vector store: ingest embeddings with NLP metadata, and run filtered semantic search."""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import MODEL_NAME, load_model  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chroma"
CORPUS_FILE = PROCESSED / "clean_corpus.jsonl"
EMB_FILE = PROCESSED / "embeddings.npy"
IDS_FILE = PROCESSED / "embedding_ids.json"
DOC_TOPICS_FILE = PROCESSED / "doc_topics.jsonl"
DOC_ENTITIES_FILE = PROCESSED / "doc_entities.jsonl"
TOPIC_INFO_FILE = PROCESSED / "topic_info.csv"
TOPIC_NAMES_FILE = PROCESSED / "topic_names.json"

COLLECTION_NAME = "arxiv_papers"
BATCH_SIZE = 1000
DEFAULT_MIN_SCORE = 0.0
_ALLOWED_META_TYPES = (str, int, float, bool)


# ---------- setup ----------

def get_client(path: Path = CHROMA_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.ClientAPI, name: str = COLLECTION_NAME,
                   reset: bool = False):
    """Get (or create) the collection. reset=True deletes it first."""
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # it didn't exist yet

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=None,  # we always pass our own embeddings
        metadata={"embedding_model": MODEL_NAME},
    )

    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != MODEL_NAME:
        raise ValueError(
            f"Collection was built with '{stored_model}' but the current model is "
            f"'{MODEL_NAME}'. Re-run ingest with --reset."
        )
    return collection


def stored_dim(collection) -> int | None:
    """Size of the vectors already in the collection, or None if it's empty."""
    if collection.count() == 0:
        return None
    got = collection.get(limit=1, include=["embeddings"])
    embs = got.get("embeddings")
    if embs is None or len(embs) == 0:
        return None
    return len(embs[0])


def has_nlp_metadata(collection) -> bool:
    """True if records carry topic metadata (ingested after topic modeling)."""
    if collection.count() == 0:
        return False
    metas = collection.get(limit=1, include=["metadatas"]).get("metadatas") or [{}]
    return "topic" in (metas[0] or {})


# ---------- NLP metadata ----------

def entity_key(name: str) -> str:
    """'GPT-4' -> 'ent_gpt_4'. One metadata flag per entity, easy to filter on."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    return f"ent_{slug}" if slug else ""


def load_topic_names() -> dict[int, str]:
    """Topic labels, with your better names from the manual review taking priority."""
    names = {}
    if TOPIC_INFO_FILE.exists():
        with open(TOPIC_INFO_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names[int(row["Topic"])] = row["CustomName"]
    names[-1] = "Unassigned"
    if TOPIC_NAMES_FILE.exists():
        custom = json.loads(TOPIC_NAMES_FILE.read_text(encoding="utf-8"))
        names.update({int(k): v for k, v in custom.items()})
    return names


def load_nlp_metadata() -> dict[str, dict]:
    """Topic and entity metadata per document ID, if the NLP steps have been run."""
    meta = defaultdict(dict)
    names = load_topic_names()

    if DOC_TOPICS_FILE.exists():
        with open(DOC_TOPICS_FILE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                t = int(rec["topic"])
                m = meta[rec["id"]]
                m["topic"] = t
                m["topic_label"] = names.get(t, rec["topic_label"])
                m["topic_reassigned"] = bool(rec.get("reassigned", False))
                if rec.get("topic_prob") is not None:
                    m["topic_prob"] = float(rec["topic_prob"])
    else:
        print("Note: doc_topics.jsonl not found, ingesting without topics.")

    if DOC_ENTITIES_FILE.exists():
        with open(DOC_ENTITIES_FILE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ents = rec["entities"]
                m = meta[rec["id"]]
                m["entities"] = ", ".join(f"{e['text']} ({e['label']})" for e in ents)
                m["n_entities"] = len(ents)
                for e in ents:
                    key = entity_key(e["text"])
                    if key:
                        m[key] = True
    else:
        print("Note: doc_entities.jsonl not found, ingesting without entities.")

    return meta


# ---------- ingest ----------

def load_records():
    """Load IDs, embeddings, texts and metadata (including NLP metadata), all in order."""
    for path in (CORPUS_FILE, EMB_FILE, IDS_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found. Run scripts/build_corpus.py and "
                f"src/embeddings.py first."
            )

    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    embeddings = np.load(EMB_FILE)

    corpus = {}
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            corpus[rec["id"]] = rec

    nlp = load_nlp_metadata()

    documents, metadatas = [], []
    for id_ in ids:
        if id_ not in corpus:
            raise KeyError(f"ID {id_} is in the embeddings but not in the corpus. "
                           f"Re-run src/embeddings.py.")
        rec = corpus[id_]
        documents.append(rec["text"])
        metadatas.append({
            "doc_id": rec["id"],
            "title": rec["title"],
            "n_words": rec["n_words"],
            "source_file": rec["source_file"],
            **nlp.get(id_, {}),
        })
    return ids, embeddings, documents, metadatas


def _clean_metadata(meta: dict | None) -> dict:
    """ChromaDB only accepts str, int, float and bool, and some versions reject {}."""
    cleaned = {
        str(k): v if isinstance(v, _ALLOWED_META_TYPES) else str(v)
        for k, v in (meta or {}).items()
        if v is not None
    }
    return cleaned or {"source_file": "unknown"}


def _check_embeddings(embeddings, collection) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (n, dim), got shape {arr.shape}")

    finite_rows = np.isfinite(arr).all(axis=1)
    if not finite_rows.all():
        bad = np.where(~finite_rows)[0]
        raise ValueError(f"{len(bad)} embeddings contain NaN or inf (first at row {bad[0]})")

    dim = stored_dim(collection)
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(
            f"Embedding size {arr.shape[1]} does not match the collection ({dim}). "
            f"Re-run ingest with --reset."
        )
    return arr


def ingest(collection, ids, embeddings, documents, metadatas) -> int:
    """Insert or update records in batches. Returns how many were stored."""
    n = len(ids)
    if not (n == len(embeddings) == len(documents) == len(metadatas)):
        raise ValueError(
            f"Length mismatch: {n} ids, {len(embeddings)} embeddings, "
            f"{len(documents)} documents, {len(metadatas)} metadatas"
        )
    if n == 0:
        print("Nothing to ingest.")
        return 0

    ids = [str(i) for i in ids]
    if any(not i.strip() for i in ids):
        raise ValueError("Empty IDs are not allowed.")

    embeddings = _check_embeddings(embeddings, collection)

    seen, keep = set(), []
    for i, id_ in enumerate(ids):
        if id_ not in seen:
            seen.add(id_)
            keep.append(i)
    if len(keep) < n:
        print(f"Skipped {n - len(keep)} duplicate IDs.")

    start = time.perf_counter()
    for s in range(0, len(keep), BATCH_SIZE):
        idx = keep[s:s + BATCH_SIZE]
        collection.upsert(
            ids=[ids[i] for i in idx],
            embeddings=embeddings[idx].tolist(),
            documents=[str(documents[i] or "") for i in idx],
            metadatas=[_clean_metadata(metadatas[i]) for i in idx],
        )
        print(f"Upserted {min(s + BATCH_SIZE, len(keep))}/{len(keep)}")

    total = collection.count()
    print(f"Ingest finished in {time.perf_counter() - start:.2f}s. "
          f"Collection now has {total} records.")
    if total > len(keep):
        print(f"Warning: {total - len(keep)} older records are still in the collection. "
              f"Use --reset to rebuild it cleanly.")
    return len(keep)


# ---------- search ----------

def make_where(topic: int | None = None, entity: str | None = None,
               exclude_reassigned: bool = False) -> dict | None:
    """Build a ChromaDB filter. Several conditions are combined with $and."""
    conditions = []
    if topic is not None:
        conditions.append({"topic": int(topic)})
    if entity:
        key = entity_key(entity)
        if not key:
            raise ValueError("Entity name is empty.")
        conditions.append({key: True})
    if exclude_reassigned:
        conditions.append({"topic_reassigned": False})

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def search(collection, model, query: str, k: int = 5,
           min_score: float = DEFAULT_MIN_SCORE, where: dict | None = None) -> list[dict]:
    """Return up to k results, best first, at most one per paper, optionally filtered."""
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")
    query = query.strip()
    if not query:
        raise ValueError("Query is empty.")

    count = collection.count()
    if count == 0:
        return []
    k = max(1, min(int(k), count))

    tokenizer = getattr(model, "tokenizer", None)
    limit = getattr(model, "max_seq_length", None)
    if tokenizer is not None and limit and len(tokenizer.encode(query)) > limit:
        print(f"Warning: query is longer than {limit} tokens and will be truncated.")

    query_emb = np.asarray(
        model.encode([query], normalize_embeddings=True, convert_to_numpy=True),
        dtype=np.float32,
    )
    dim = stored_dim(collection)
    if dim is not None and query_emb.shape[1] != dim:
        raise ValueError(
            f"Query vector size {query_emb.shape[1]} does not match the collection ({dim}). "
            f"The collection was probably built with a different model."
        )

    n_fetch = min(k * 4, count)
    res = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=n_fetch,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results, seen_docs = [], set()
    for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        score = 1 - dist / 2  # squared L2 on unit vectors -> cosine similarity
        if score < min_score:
            break
        meta = meta or {}
        doc_id = meta.get("doc_id", id_)
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        results.append({
            "id": id_,
            "doc_id": doc_id,
            "title": meta.get("title", ""),
            "score": round(score, 4),
            "topic": meta.get("topic"),
            "topic_label": meta.get("topic_label", ""),
            "entities": meta.get("entities", ""),
            "text": doc or "",
        })
        if len(results) == k:
            break
    return results


# ---------- command line ----------

def cmd_topics() -> None:
    if not TOPIC_INFO_FILE.exists():
        print("No topics yet. Run: python src/topics.py")
        return
    names = load_topic_names()
    with open(TOPIC_INFO_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{'Topic':>5}  {'Papers':>6}  Name")
    for row in rows:
        t = int(row["Topic"])
        print(f"{t:>5}  {row['Count']:>6}  {names.get(t, row['CustomName'])}")


def main():
    parser = argparse.ArgumentParser(description="ChromaDB ingest and search.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="load embeddings and metadata into ChromaDB")
    p_ingest.add_argument("--reset", action="store_true", help="delete the collection first")

    sub.add_parser("topics", help="list topics you can filter on")

    p_search = sub.add_parser("search", help="search the collection")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    p_search.add_argument("--topic", type=int, help="only this topic number")
    p_search.add_argument("--entity", help="only papers mentioning this entity, e.g. Qwen3")
    p_search.add_argument("--no-reassigned", action="store_true",
                          help="skip papers whose topic came from outlier reassignment")

    args = parser.parse_args()

    if args.command == "topics":
        cmd_topics()
        return

    client = get_client()

    if args.command == "ingest":
        collection = get_collection(client, reset=args.reset)
        ingest(collection, *load_records())
        return

    # search
    collection = get_collection(client)
    filtering = args.topic is not None or args.entity or args.no_reassigned

    if filtering and not has_nlp_metadata(collection):
        print("This collection has no topic/entity metadata yet. "
              "Run: python src/vector_store.py ingest --reset")
        return
    if args.topic is not None and TOPIC_INFO_FILE.exists():
        known = set(load_topic_names())
        if args.topic not in known:
            print(f"Topic {args.topic} does not exist. "
                  f"Run: python src/vector_store.py topics")
            return

    where = make_where(args.topic, args.entity, args.no_reassigned)
    model = load_model()
    start = time.perf_counter()
    results = search(collection, model, args.query, args.k, args.min_score, where)
    elapsed_ms = 1000 * (time.perf_counter() - start)

    if not results:
        if collection.count() == 0:
            print("No results. The collection is empty; run ingest first.")
        elif where:
            print(f"No results match this filter: {where}")
        else:
            print(f"No results above score {args.min_score}.")
        return

    filter_note = f"  filter: {where}" if where else ""
    print(f"\nTop {len(results)} for: \"{args.query}\"  ({elapsed_ms:.1f} ms){filter_note}\n")
    for rank, r in enumerate(results, 1):
        title = r["title"]
        snippet = (r["text"][len(title):].strip()
                   if title and r["text"].startswith(title) else r["text"])
        print(f"{rank}. [{r['score']:.3f}] {title}")
        if r["topic"] is not None:
            print(f"   Topic {r['topic']}: {r['topic_label']}")
        if r["entities"]:
            print(f"   Entities: {r['entities'][:120]}")
        print(f"   {snippet[:200]}...\n")


if __name__ == "__main__":
    main()