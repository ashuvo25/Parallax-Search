
# Chunking Strategy and Vector Store

Parallax-Search, Week 2

|                 |                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------- |
| Components      | `src/chunking.py` (chunker), `src/vector_store.py` (ChromaDB)                         |
| Embedding model | `all-MiniLM-L6-v2` (384 dimensions, 256 token limit)                                    |
| Vector store    | ChromaDB, persistent, stored in `data/chroma/`                                          |
| Status          | Vector store implemented. Chunker specified in this document, implementation in progress. |

## 1. Purpose

Documents are embedded with `all-MiniLM-L6-v2`, which reads at most 256 tokens. Anything longer is cut off silently, so the end of a long abstract never reaches the embedding and cannot be found by search. Chunking splits each document into pieces that fit inside that limit, so every part of the text is embedded and searchable.

This document defines how documents are chunked, how chunks are stored in ChromaDB, and how the vector store handles invalid input and failure cases.

## 2. Chunking strategy

### 2.1 Method

Recursive splitting, measured in tokens.

The chunker tries to split text at the most natural boundary first and only falls back to a finer boundary when a piece is still too long:

| Level | Separator | Purpose                   |
| ----- | --------- | ------------------------- |
| 1     | `\n\n`  | paragraph                 |
| 2     | `\n`    | line                      |
| 3     | `. `    | sentence                  |
| 4     |           | word                      |
| 5     | none      | hard split on token count |

The cleaned corpus has all line breaks collapsed (see `clean_whitespace` in `src/preprocessing.py`), so in practice splitting happens at sentence level, then word level. The paragraph and line levels are kept so the chunker works unchanged if a future source keeps paragraph structure.

Sizes are counted with the embedding model's own tokenizer, not in characters or words. This is the only way to guarantee a chunk fits the model's limit, since the number of tokens per word varies a lot in technical text (formulas, acronyms, model names).

### 2.2 Parameters

| Parameter            | Value           | Reason                                                                                               |
| -------------------- | --------------- | ---------------------------------------------------------------------------------------------------- |
| `CHUNK_SIZE`       | 200 tokens      | Leaves room under the 256 token limit for the title prefix and the two special tokens the model adds |
| `CHUNK_OVERLAP`    | 40 tokens (20%) | A sentence cut at a chunk boundary still appears whole in one of the two chunks                      |
| `MIN_CHUNK_SIZE`   | 50 tokens       | A final piece smaller than this is merged into the previous chunk                                    |
| `MODEL_MAX_TOKENS` | 254             | 256 minus the `[CLS]` and `[SEP]` tokens. No chunk may exceed this.                              |

All parameters are constants at the top of `src/chunking.py`. Changing any of them requires re-running embeddings and re-ingesting with `--reset` (section 4.5).

### 2.3 Rules

1. **Short documents are not split.** If the whole text fits in `CHUNK_SIZE`, the document becomes a single chunk. Most arXiv abstracts fall in this case.
2. **The title is prefixed to every chunk.** Each chunk's text is `"{title}. {chunk body}"`. Without this, a chunk from the middle of an abstract has no context about what the paper is, and its embedding is weaker. The title's tokens count toward the chunk size, so the body budget per chunk is `CHUNK_SIZE` minus the title length.
3. **Long titles are capped.** If a title is longer than half of `CHUNK_SIZE`, only the first 100 tokens of it are used as the prefix, so every chunk keeps at least 100 tokens for the body.
4. **Overlap is applied at sentence level.** The next chunk starts with the last whole sentences of the previous chunk, up to `CHUNK_OVERLAP` tokens. Overlap never starts in the middle of a sentence unless a single sentence is longer than the overlap.
5. **A small final piece is merged.** If the last chunk has fewer than `MIN_CHUNK_SIZE` tokens and merging it into the previous chunk stays within `MODEL_MAX_TOKENS`, it is merged. Otherwise it is kept as its own chunk.
6. **Chunking is deterministic.** The same document and parameters always produce the same chunks and the same IDs.

### 2.4 Chunk IDs and metadata

Chunk ID format:

```
{doc_id}_c{chunk_index:03d}
```

For example `3f9a1c2b7d4e_c000`, `3f9a1c2b7d4e_c001`. The `doc_id` is the document's ID from `clean_corpus.jsonl`.

Each chunk stores this metadata in ChromaDB:

| Field           | Type | Description                                               |
| --------------- | ---- | --------------------------------------------------------- |
| `doc_id`      | str  | ID of the paper the chunk belongs to                      |
| `chunk_index` | int  | Position of the chunk in the paper, starting at 0         |
| `n_chunks`    | int  | Total number of chunks for this paper                     |
| `title`       | str  | Paper title                                               |
| `n_tokens`    | int  | Token count of the chunk text, including the title prefix |
| `start_char`  | int  | Start position of the chunk body in the original text     |
| `end_char`    | int  | End position of the chunk body in the original text       |
| `source_file` | str  | PDF the paper came from                                   |

`start_char` and `end_char` make it possible to highlight the matching passage in the full abstract.

### 2.5 Chunker edge cases

| Input                                                                      | Behavior                                                                   |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Empty or whitespace-only text                                              | Returns no chunks. The document is skipped and counted in the run summary. |
| Text shorter than `CHUNK_SIZE`                                           | One chunk containing the whole text                                        |
| Single sentence longer than `CHUNK_SIZE`                                 | Split at word level                                                        |
| Single word longer than `CHUNK_SIZE` (for example a long formula or URL) | Hard split by token count                                                  |
| Missing or empty title                                                     | Chunk text is the body only, with no prefix                                |
| `CHUNK_OVERLAP >= CHUNK_SIZE`                                            | `ValueError` at startup, since chunking would never advance              |
| `CHUNK_SIZE > MODEL_MAX_TOKENS`                                          | `ValueError` at startup, since chunks would be truncated                 |

### 2.6 Alternatives considered

| Approach                                                   | Reason not used                                                                                                                          |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| No chunking                                                | Text past 256 tokens is truncated and never searchable                                                                                   |
| Fixed-size character splitting                             | Cuts words and sentences in half, and character counts do not map reliably to tokens                                                     |
| Sentence-only splitting                                    | Chunk sizes vary widely. One long sentence can still exceed the token limit.                                                             |
| Semantic chunking (split where embedding similarity drops) | Needs an extra embedding pass per sentence and is harder to test. The gain is small on abstracts, which are short and on a single topic. |
| Larger chunks with a long-context model                    | Larger models are several times slower on CPU, and the current latency budget is met with the small model                                |

### 2.7 Testing

Unit tests live in `tests/test_chunking.py` and must cover:

- every chunk is at or below `MODEL_MAX_TOKENS`
- a short document produces exactly one chunk
- consecutive chunks overlap
- joining the chunk bodies, with overlaps removed, gives back the original text
- every edge case in section 2.5
- the same input always produces the same chunks and IDs

## 3. Vector store

### 3.1 Setup

| Setting             | Value                                                                        |
| ------------------- | ---------------------------------------------------------------------------- |
| Client              | `chromadb.PersistentClient`                                                |
| Storage path        | `data/chroma/` (git-ignored)                                               |
| Collection          | `arxiv_papers`                                                             |
| Collection metadata | `{"embedding_model": "all-MiniLM-L6-v2"}`                                  |
| Embedding function  | none. Embeddings are always computed by `src/embeddings.py` and passed in. |
| Distance            | ChromaDB default, squared L2                                                 |
| Telemetry           | disabled                                                                     |

Embeddings are normalized to unit length before they are stored. For unit vectors, squared L2 distance and cosine similarity are directly related:

```
cosine_similarity = 1 - distance / 2
```

So the ranking is identical to a cosine index, and the score shown to users is the cosine similarity, between about 0 and 1. The default distance is used on purpose, because the way to configure a cosine index differs between ChromaDB versions.

ChromaDB never computes embeddings itself. This guarantees documents and queries always go through the same model.

### 3.2 Ingest

```bash
python src/vector_store.py ingest           # add or update records
python src/vector_store.py ingest --reset   # delete the collection and rebuild it
```

Ingest reads `clean_corpus.jsonl`, `embeddings.npy` and `embedding_ids.json`, validates them (section 3.4), and writes records in batches of 1,000 using `upsert`. Running ingest twice with the same data leaves the collection unchanged.

### 3.3 Search

```bash
python src/vector_store.py search "query text" -k 5 --min-score 0.3
```

1. The query is validated, then embedded with the same model and normalized.
2. ChromaDB is asked for `k × 4` results (capped at the collection size).
3. Results below `--min-score` are dropped.
4. Only the best-scoring chunk of each paper (`doc_id`) is kept.
5. The top `k` papers are returned, best first.

Step 2 fetches extra results because once documents are chunked, several chunks of the same paper can rank at the top. Without step 4, one long paper could fill most of the result list.

### 3.4 Edge cases

**Ingest**

| Case                                               | Handling                                                                                           |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Input files missing                                | `FileNotFoundError` naming the script to run first                                               |
| ID, embedding, document and metadata counts differ | `ValueError` before anything is written                                                          |
| Nothing to ingest                                  | Returns 0, writes nothing                                                                          |
| Empty or whitespace ID                             | `ValueError`                                                                                     |
| Duplicate IDs in the input                         | First occurrence kept, the rest skipped and reported. ChromaDB rejects duplicates within one call. |
| Same data ingested twice                           | `upsert` updates the existing records, so no duplicates                                          |
| Embeddings not 2D                                  | `ValueError`                                                                                     |
| NaN or infinite values in an embedding             | `ValueError` with the first bad row                                                              |
| Embedding size differs from vectors already stored | `ValueError` suggesting `--reset`                                                              |
| Metadata value is `None`                         | Field dropped                                                                                      |
| Metadata value is a list, dict or other type       | Converted to a string. ChromaDB only accepts str, int, float and bool.                             |
| Metadata is empty                                  | Placeholder `source_file: "unknown"` added, since some ChromaDB versions reject `{}`           |
| Document text is `None`                          | Stored as an empty string                                                                          |
| Large input                                        | Written in batches of 1,000, below ChromaDB's per-call limit                                       |
| Records from a previous run no longer in the input | Kept by `upsert`. A warning is printed suggesting `--reset`.                                   |
| ID in embeddings but not in the corpus             | `KeyError` asking to re-run `src/embeddings.py`                                                |

**Collection**

| Case                                              | Handling                                       |
| ------------------------------------------------- | ---------------------------------------------- |
| Collection does not exist                         | Created on first use                           |
| `--reset` when the collection does not exist    | Ignored, then a new collection is created      |
| Collection built with a different embedding model | `ValueError` on load, suggesting `--reset` |

**Search**

| Case                                          | Handling                                                              |
| --------------------------------------------- | --------------------------------------------------------------------- |
| Query is not a string                         | `TypeError`                                                         |
| Query is empty or whitespace                  | `ValueError`                                                        |
| Query longer than 256 tokens                  | Warning that it will be truncated                                     |
| Collection is empty                           | Returns an empty list                                                 |
| `k` is 0 or negative                        | Raised to 1                                                           |
| `k` larger than the collection              | Lowered to the collection size                                        |
| Query vector size differs from the collection | `ValueError` explaining the collection was built with another model |
| Several chunks of one paper match             | Best chunk per paper kept                                             |
| No result above `--min-score`               | Empty list, with a message in the command line                        |

### 3.5 Operations

| Situation                   | Action                                                                                                               |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| First setup                 | Run embeddings, then `ingest`                                                                                      |
| Corpus rebuilt              | Re-run `src/embeddings.py`, then `ingest --reset`                                                                |
| Embedding model changed     | Change `MODEL_NAME`, re-run embeddings, then `ingest --reset`. Search refuses to run on a mismatched collection. |
| Chunking parameters changed | Re-chunk, re-run embeddings, then `ingest --reset`                                                                 |
| Corrupted store             | Delete `data/chroma/` and run `ingest`. It is fully rebuilt from the files in `data/processed/`.               |

`data/chroma/` is derived data and is never edited by hand.

## 4. Performance baseline

Measured with `scripts/test_retrieval.py` on 2026-10-18, before chunking, with one record per paper.

|         |                     |
| ------- | ------------------- |
| Records | 5,007               |
| Device  | CPU                 |
| k       | 5, 5 runs per query |

**Latency**

| Metric               | Value                   |
| -------------------- | ----------------------- |
| p50                  | 14.61 ms                |
| p95                  | 18.21 ms                |
| Max                  | 20.32 ms                |
| Query encoding share | about 80% of total time |
| ChromaDB query       | about 2 ms              |

Latency does not change meaningfully between k = 1 and k = 100 at this collection size.

**Retrieval sanity check** (paper title as query, 200 random papers)

| Metric   | Value |
| -------- | ----- |
| Recall@1 | 0.985 |
| Recall@5 | 1.000 |
| MRR      | 0.990 |

**Score ranges observed**

| Query type                                            | Top score    |
| ----------------------------------------------------- | ------------ |
| Relevant queries                                      | 0.44 to 0.64 |
| Meaningless queries (`a`, `????`, stopwords only) | 0.17 to 0.21 |

A `--min-score` of 0.3 separates these two groups. After chunking, this benchmark must be re-run and compared against these numbers.

## 5. Known limitations

- **Duplicated titles in some documents.** Titles that contained characters not allowed in Windows file names (such as `:` or `?`) appear twice in the document text, once without the character and once with it. This comes from the preprocessing step, which took the title from the file name. A fix that reads the title from the PDF's font size is ready and will be applied before the chunked rebuild.
- **The score threshold is not a language filter.** A French query scored 0.44 with an unrelated result, which is above the 0.3 threshold. The model is English-only, and so is the corpus.
- **Single writer only.** `PersistentClient` is meant for one process. Running two ingests at the same time can corrupt the store. Concurrent access would need ChromaDB in client and server mode.
- **The quality check is a sanity check, not a benchmark.** The title is part of each document's text, so title queries are expected to score close to 100%. A real quality benchmark needs queries with human-labeled relevant papers.
