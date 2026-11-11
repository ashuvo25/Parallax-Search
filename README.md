# Parallax-Search

This project collects 5,000 recent AI papers from arXiv, cleans them into a text dataset, embeds them with sentence-transformers and stores them in ChromaDB for semantic search. Questions are answered by an LLM using the papers that the search finds. Each paper is also tagged with a topic and its named entities, so searches can be filtered. The system is served as a FastAPI application and evaluated with retrieval and end-to-end test sets. Each paper is stored as one document containing its title and abstract.

## Project structure

```
Internship-project-Parallax-Labs/
├── data/
│   ├── raw/                          PDFs, one per paper
│   │   └── rejected/                 papers removed by the language filter
│   ├── processed/
│   │   ├── clean_corpus.jsonl        cleaned dataset
│   │   ├── embeddings.npy            document vectors
│   │   ├── embedding_ids.json        document IDs, same order as the vectors
│   │   ├── doc_topics.jsonl          topic of each paper
│   │   ├── topic_info.csv            list of topics with their top words
│   │   ├── topic_names.json          topic names from the manual review
│   │   └── doc_entities.jsonl        named entities of each paper
│   └── chroma/                       ChromaDB database
├── docs/
│   ├── chunking_and_vector_store.md
│   └── nlp_metadata.md
├── eval/
│   ├── ner_gold.csv                  hand-labeled entities for 50 papers
│   ├── topic_review.csv              manual review of each topic
│   ├── retrieval_queries.csv         20 test queries for retrieval
│   ├── retrieval_judgments.csv       relevance labels for the retrieval test
│   └── e2e_questions.csv             20 test questions for end-to-end evaluation
├── logs/
│   ├── api.jsonl                     one JSON line per API request
│   ├── embeddings.log
│   ├── retrieval.log
│   ├── retrieval_report.json
│   ├── retrieval_eval.json
│   ├── retrieval_eval_per_query.csv
│   ├── e2e_eval.json
│   ├── e2e_eval_results.csv
│   ├── e2e_eval_answers.jsonl
│   ├── topics.log
│   ├── topic_metrics.json
│   ├── topic_review_summary.json
│   ├── ner_eval_baseline.json
│   └── ner_eval_with_rules.json
├── models/
│   └── bertopic/                     saved topic model
├── reports/
│   ├── topics_barchart.html
│   └── topics_map.html
├── scripts/
│   ├── verify_env.py
│   ├── fetch_arxiv.py
│   ├── build_corpus.py
│   ├── test_retrieval.py
│   ├── eval_retrieval.py
│   └── eval_e2e.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── prompts.py
│   ├── rag.py
│   ├── topics.py
│   ├── topic_quality.py
│   ├── ner.py
│   └── api/
│       ├── __init__.py
│       ├── main.py                   app and endpoints
│       ├── schemas.py                request and response models
│       ├── errors.py                 error format and handlers
│       ├── middleware.py             request logging and request IDs
│       └── logging_config.py         JSON log setup
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
├── .env.example
├── requirements.txt
└── README.md
```

The `data/`, `logs/` and `models/` folders, `.env` and `venv/` are in `.gitignore`, so none of them are in the repository. Run the steps below to generate them. The `eval/` folder is committed, because it holds test sets and labels that cannot be regenerated.

## Setup

Python 3.10 or newer is required. The scripts were written and run on Windows.

Create and activate a virtual environment from the project root:

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
venv\Scripts\activate.bat        # CMD
source venv/bin/activate         # macOS / Linux
```

Install PyTorch first. For a CUDA build, get the install command from https://pytorch.org/get-started/locally/. Otherwise:

```bash
pip install torch
```

Then install the rest:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### API key

Answer generation needs an API key from an LLM provider. The default is Groq, which has a free tier. Create a key at https://console.groq.com/keys, then copy `.env.example` to `.env` and fill it in:

```
LLM_API_KEY=gsk_your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
LLM_REASONING_EFFORT=low
```

`.env` is never committed. `.env.example` lists the same settings without a key.

## Running the pipeline

Run these from the project root with the virtual environment active, in this order.

### 1. Check the environment

```bash
python scripts/verify_env.py
```

Prints the PyTorch version and whether CUDA is available. A CPU-only machine will print "CUDA is not available", which is fine.

### 2. Download the papers

```bash
python scripts/fetch_arxiv.py
```

Uses the arXiv API to get the 5,000 most recent papers from cs.AI, cs.CL and cs.LG. For each paper it creates a one-page PDF with the title and abstract and saves it to `data/raw/`, named after the paper title. Characters that Windows does not allow in file names are removed, and long titles are cut to 150 characters. If two papers have the same title, the arXiv ID is added to the second file name.

This takes a few minutes. Files that already exist are skipped, so the script can be run again if it stops.

### 3. Run the tests

```bash
python -m pytest tests -v
```

Tests the cleaning functions and every API endpoint. All tests should pass before moving on.

### 4. Clean the PDFs

Back up the raw data first, since this step overwrites the files:

```powershell
Copy-Item data\raw data\raw_backup -Recurse
```

Then run:

```bash
python src/preprocessing.py
```

Each PDF in `data/raw/` is read, cleaned and saved again under the same name. Papers that are not in English are moved to `data/raw/rejected/`. A summary of how many files were cleaned, rejected or failed is printed at the end.

### 5. Build the dataset

```bash
python scripts/build_corpus.py
```

Reads the cleaned PDFs, checks each document and writes the final dataset to `data/processed/clean_corpus.jsonl`.

### 6. Generate embeddings

```bash
python src/embeddings.py
```

Embeds the `text` field of every document with `all-MiniLM-L6-v2` and saves the vectors to `data/processed/embeddings.npy` and the matching IDs to `data/processed/embedding_ids.json`. The model is downloaded on the first run (about 90 MB) and cached after that.

To try several batch sizes first and use the fastest one:

```bash
python src/embeddings.py --benchmark
```

### 7. Find topics

```bash
python src/topics.py
```

Groups the papers into topics with BERTopic, reusing the saved embeddings. Writes the topic of each paper to `data/processed/doc_topics.jsonl`, the topic list to `data/processed/topic_info.csv`, the model to `models/bertopic/` and two charts to `reports/`. Takes about a minute on CPU.

Options:

```bash
python src/topics.py --min-cluster-size 10       # allow smaller topics
python src/topics.py --nr-topics 30              # merge similar topics down to 30
python src/topics.py --outlier-threshold 0.6     # stricter outlier reassignment (0 turns it off)
```

### 8. Check topic quality

```bash
python src/topic_quality.py metrics
python src/topic_quality.py review
```

`metrics` prints and saves automatic quality numbers. `review` creates `eval/topic_review.csv` for checking each topic by hand. Fill in the `coherent` column (`yes`, `partly` or `no`) and optionally `better_name` and `notes`, then run:

```bash
python src/topic_quality.py summarize
```

### 9. Extract named entities

```bash
python src/ner.py run
```

Finds organizations, models and datasets, people, places and languages in every paper, and writes them to `data/processed/doc_entities.jsonl`. Takes about two minutes on CPU.

To measure accuracy, create a sample of 50 papers to label by hand (once):

```bash
python src/ner.py sample
```

Fill in the `gold_entities` column of `eval/ner_gold.csv` in the form `OpenAI|ORG; GPT-4|PRODUCT`, or `NONE` for a paper without entities. Then compare plain spaCy with the improved version:

```bash
python src/ner.py evaluate --baseline
python src/ner.py evaluate
```

### 10. Load everything into ChromaDB

```bash
python src/vector_store.py ingest --reset
```

Stores every paper's ID, vector, text and metadata, including its topic and entities, in the ChromaDB database in `data/chroma/`. Run it again with `--reset` whenever the embeddings, topics, entities or topic names change. It takes under a minute.

### 11. Search from the command line

```bash
python src/vector_store.py search "making large language models safer"
python src/vector_store.py search "graph neural networks for molecules" -k 10 --min-score 0.3
python src/vector_store.py topics
python src/vector_store.py search "reward models for alignment" --topic 0
python src/vector_store.py search "open-weight model evaluation" --entity Qwen3
python src/vector_store.py search "tool use" --topic 0 --no-reassigned
```

`-k` sets the number of results, `--min-score` hides results below a score, `--topic` and `--entity` filter the results, and `--no-reassigned` leaves out papers whose topic came from outlier reassignment.

### 12. Check the LLM connection

```bash
python src/llm.py --list-models
python src/llm.py --ping
```

`--list-models` prints the model IDs that are live for your API key and warns if `LLM_MODEL` is not one of them. Free model IDs change often, so run this first whenever a request fails with "model not found". `--ping` sends a short test request.

### 13. Ask questions from the command line

```bash
python src/rag.py "What methods are used to make large language models safer?"
python src/rag.py        # interactive mode
```

Each answer is followed by its source papers, the retrieval, generation and total time, and the token usage.

### 14. Start the API

```bash
uvicorn src.api.main:app --reload --no-access-log
```

Open http://127.0.0.1:8000/docs in a browser to try every endpoint. `--no-access-log` turns off uvicorn's own request log, since the app writes its own structured log.

### 15. Evaluate retrieval

```bash
python scripts/eval_retrieval.py pool
python scripts/eval_retrieval.py evaluate
```

`pool` collects candidate papers for each test query into `eval/retrieval_judgments.csv` for relevance labeling. `evaluate` computes Precision@K, Recall@K and MRR for semantic search and a keyword baseline.

### 16. Evaluate end to end

```bash
python scripts/eval_e2e.py
python scripts/eval_e2e.py --judge
python scripts/eval_e2e.py --api http://127.0.0.1:8000
```

Runs the 20 test questions through the full system and measures answer behavior, citations, groundedness, relevance and latency. `--judge` adds LLM-as-judge scores, and `--api` sends the questions through the running API instead of calling the pipeline directly.

## Text cleaning

The functions are in `src/preprocessing.py` and are applied in this order:

1. `strip_html` removes HTML tags and decodes entities such as `&amp;`. It only parses the text as HTML when it finds a real tag like `<p>` or `<br>`, so math in abstracts such as `x<y` is left alone.
2. `normalize_unicode` applies NFKC normalization, which turns ligatures like `ﬁ` into `fi` and non-breaking spaces into normal spaces. It also removes zero-width and control characters.
3. `clean_whitespace` collapses line breaks, tabs and repeated spaces into single spaces.
4. `detect_language` identifies the language with langdetect. Only English documents are kept.

`clean_text` runs the first three steps together.

## Dataset

The cleaned dataset is at `data/processed/clean_corpus.jsonl`. It has one JSON object per line, one line per paper:

| Field           | Description                                          |
| --------------- | ---------------------------------------------------- |
| `id`          | 12-character hash of the text, the same on every run |
| `title`       | paper title                                          |
| `abstract`    | paper abstract                                       |
| `text`        | title and abstract together                          |
| `lang`        | detected language, always `en`                     |
| `n_words`     | number of words in `text`                          |
| `source_file` | name of the PDF in `data/raw/`                     |

A document is left out of the dataset if it is empty, has fewer than 20 words, is not in English, would still change if cleaned again, or has the same text as a document already added.

## Embeddings

`src/embeddings.py` uses `all-MiniLM-L6-v2` from sentence-transformers. It produces 384-dimensional vectors, runs well on CPU and reads up to 256 tokens per text. Vectors are normalized to unit length so that similarity can be compared with a simple dot product. Each run logs the model load time, how many texts are truncated, and the total time, texts per second and milliseconds per text to `logs/embeddings.log`.

## Topics

`src/topics.py` uses BERTopic. It reduces the saved embeddings to 5 dimensions with UMAP, groups them with HDBSCAN, and describes each group with its most distinctive words. Words that appear in almost every abstract are removed first, and the random seed is fixed.

Outliers are moved to their most similar topic when the similarity is at least 0.5 and marked as reassigned. Papers under 60 words are marked as short, and words that appear among the top words of many topics are reported as stop word candidates.

| Metric                               | Value                      |
| ------------------------------------ | -------------------------- |
| Topics                               | 42                         |
| Outliers before / after reassignment | 1,622 (32.4%) / 201 (4.0%) |
| Topic diversity                      | 0.729                      |
| Mean cohesion (whole corpus: 0.517)  | 0.666                      |

## Named entities

`src/ner.py` uses spaCy's `en_core_web_sm` model and keeps five entity types: `ORG`, `PRODUCT` (named models, datasets and benchmarks), `PERSON`, `GPE` and `LANGUAGE`. Rules for known model families, datasets and companies, a list of jargon terms that are never entities, and a filter for lowercase spans were added because plain spaCy mislabeled AI terms. With these changes, `PRODUCT` entities went from 879 to 1,997. Details are in `docs/nlp_metadata.md`.

## Vector store

`src/vector_store.py` stores the embeddings in a persistent ChromaDB collection named `arxiv_papers` in `data/chroma/`. Documents and queries are embedded with the same model in our own code, and the collection records which model it was built with. Each record holds the document ID, vector, text, and metadata including topic, entities and one true/false flag per entity for filtering.

The collection uses ChromaDB's default squared L2 distance. Because the vectors are normalized, this gives the same ranking as cosine similarity, and the score shown is the cosine similarity, calculated as `1 - distance / 2`. Records are written with `upsert` in batches of 1,000, and input is checked before anything is written. The full list of edge cases is in `docs/chunking_and_vector_store.md` and `docs/nlp_metadata.md`.

## Answer generation

Questions are answered in three steps:

1. **Retrieve.** The question is embedded and the five most similar papers are fetched from ChromaDB. Papers scoring below 0.3 are dropped.
2. **Build the prompt.** The papers are numbered and placed in the prompt as context, together with the question.
3. **Generate.** The LLM writes an answer based on that context and cites the papers it used as `[1]`, `[2]` and so on.

If no paper scores above 0.3, the LLM is not called and the answer says that no relevant papers were found.

The client in `src/llm.py` uses the `openai` Python package, which works with any provider that offers an OpenAI-compatible API, so switching between Groq and OpenRouter only means changing `.env`. The default model is `openai/gpt-oss-20b` on Groq.

The client times out after 30 seconds and retries rate limits, timeouts and server errors twice with increasing waits, respecting the provider's `Retry-After` header. Errors that remain after the retries are turned into specific error types, each mapped to an HTTP status code by the API.

| Setting                  | Default                            | Description                                            |
| ------------------------ | ---------------------------------- | ------------------------------------------------------ |
| `LLM_API_KEY`          | none                               | API key for the provider                               |
| `LLM_BASE_URL`         | `https://api.groq.com/openai/v1` | Provider's API address                                 |
| `LLM_MODEL`            | `openai/gpt-oss-20b`             | Model ID                                               |
| `LLM_REASONING_EFFORT` | `low`                            | `low`, `medium`, `high`, or empty to turn it off |
| `LLM_MAX_TOKENS`       | `1024`                           | Maximum length of the answer, including reasoning      |
| `LLM_TEMPERATURE`      | `0.2`                            | Kept low so answers stay close to the context          |
| `LLM_TIMEOUT`          | `30`                             | Seconds before a request times out                     |
| `LLM_MAX_RETRIES`      | `2`                              | Automatic retries before an error is returned          |

## API

The FastAPI application in `src/api/` loads the embedding model, the ChromaDB collection and the LLM client once at startup. If no API key is configured, the API still starts: search works and `/ask` returns 503.

### Endpoints

| Method   | Path        | Description                                                                        |
| -------- | ----------- | ---------------------------------------------------------------------------------- |
| `GET`  | `/health` | Status, number of documents, model names, whether the LLM is configured            |
| `GET`  | `/topics` | Topics that can be used as filters                                                 |
| `POST` | `/search` | Semantic search, with optional topic and entity filters. No LLM call.              |
| `POST` | `/ask`    | Answer a question from the retrieved papers, with sources, timings and token usage |

Example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" `
  -Body '{"question": "How are LLM agents evaluated?"}'
```

Request bodies are validated with Pydantic. `question` and `query` are limited to 1,000 characters, `k` to 1–10 for `/ask` and 1–50 for `/search`.

### Errors

Every error uses the same format:

```json
{"error": {"code": 400, "type": "bad_request", "message": "Question must not be blank.", "request_id": "a1b2c3d4e5f6"}}
```

| Code | When                                                                                    |
| ---- | --------------------------------------------------------------------------------------- |
| 400  | Blank question or query, unknown topic, blank entity, body that is not valid JSON       |
| 404  | Unknown path                                                                            |
| 405  | Wrong HTTP method                                                                       |
| 422  | Missing field, wrong type or value out of range. The `details` list names each field. |
| 429  | The LLM provider's rate limit was reached. A `Retry-After` header is included.        |
| 500  | Unexpected error. The user gets a generic message, the full error goes only to the log. |
| 503  | LLM not configured or unreachable, or filters used before topic metadata was loaded     |
| 504  | The LLM did not answer within the timeout                                               |

### Logging

Every request writes one JSON line to `logs/api.jsonl` and the console, with the request ID, method, path, status, latency and client address. `/search` and `/ask` add details such as the number of results, retrieval and generation time and token counts. The question text itself is not logged, only its length. The log file rotates at 5 MB and keeps three old files.

Each request gets an ID that is returned in the `X-Request-ID` response header and included in every error. A client can also send its own `X-Request-ID`, which is then used instead.

Example log line:

```json
{"time": "2026-11-11T16:20:31.412+00:00", "level": "INFO", "logger": "api.requests", "message": "POST /ask 200", "request_id": "a1b2c3d4e5f6", "method": "POST", "path": "/ask", "status": 200, "latency_ms": 2104.6, "question_chars": 29, "k": 5, "llm_called": true, "sources": 5, "retrieval_ms": 21.3, "generation_ms": 2071.8, "prompt_tokens": 1852, "completion_tokens": 512}
```

## Tests

`tests/test_api.py` tests every endpoint with FastAPI's `TestClient`. The embedding model, ChromaDB and the LLM are replaced with fakes, so the tests run in seconds, cost nothing and give the same result every time. The fakes also make it possible to force errors that are hard to trigger by hand, such as a rate limit, a timeout or an unexpected crash.

The tests cover successful requests, every error code listed above, that internal error details never reach the user, request ID handling, and that each request produces one structured log line without the question text.

`tests/test_preprocessing.py` tests the text cleaning functions.

## Evaluation

### Retrieval

`scripts/eval_retrieval.py` measures retrieval on 20 test queries covering the main topics of the corpus.

For each query, the top 10 results of semantic search and the top 10 of a TF-IDF keyword search are pooled, about 17 papers per query and 333 in total, and labeled relevant or not. Pooling two different methods means recall is not measured only against papers our own system found. Candidates are shuffled and shown without their rank or source. The labels were drafted with an LLM and should be checked by hand on a sample.

| Metric       | Semantic search | Keyword (TF-IDF) |
| ------------ | --------------- | ---------------- |
| Precision@1  | 0.750           | 0.700            |
| Precision@5  | 0.630           | 0.660            |
| Precision@10 | 0.525           | 0.595            |
| Recall@5     | 0.396           | 0.433            |
| Recall@10    | 0.623           | 0.718            |
| MRR          | 0.842           | 0.850            |

Semantic search ranks the first result best, but keyword search is slightly ahead at deeper cut-offs. The test queries use technical terms that appear word for word in abstracts, `all-MiniLM-L6-v2` is a small general-purpose model, and long abstracts are cut at 256 tokens. The two methods agreed on only about 3 of 10 papers per query, which suggests that combining them would beat both. The hardest queries were retrieval-augmented generation (q13), LLM agent evaluation (q01) and multilingual models (q09).

### End to end

`scripts/eval_e2e.py` runs 20 questions through the full system. Each question has an expected behavior:

| Group         | Questions                                                                                      | Expected                                                        |
| ------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| In domain     | 12 questions on topics in the corpus                                                           | Answer with citations                                           |
| Off topic     | 5, such as "What is the capital of France?" or "Write a poem about the ocean."                 | Refuse                                                          |
| Out of corpus | 3 about AI papers that are not in the corpus, such as the original "Attention Is All You Need" | Refuse, since the system must answer only from retrieved papers |

It measures:

- **Behavior accuracy**: whether the system answered or refused as expected
- **Citation rate** and **invalid citations**: whether answers cite sources, and only sources that exist
- **Groundedness**: the share of answer sentences whose closest sentence in the retrieved papers has a similarity of at least 0.5
- **Answer relevance**: similarity between the question and the answer
- **LLM judge** (optional): faithfulness and relevance scores from 1 to 5
- **Latency and tokens**: retrieval, generation and total time

The script tracks the tokens used in the last minute and waits when the next call would exceed the free tier's limit of 8,000 tokens per minute. Every answer, with the sentences that failed the groundedness check, is saved to `logs/e2e_eval_answers.jsonl`.

Results on CPU:

| Metric                                | Value                      |
| ------------------------------------- | -------------------------- |
| Behavior accuracy                     | 0.85 (17 of 20)            |
| In domain / off topic / out of corpus | 12 of 12 / 4 of 5 / 1 of 3 |
| Citation rate                         | 0.67                       |
| Invalid citations                     | 0                          |
| Groundedness                          | 0.76                       |
| Answer relevance                      | 0.63                       |
| LLM judge faithfulness / relevance    | 4.27 / 4.47 out of 5       |
| Total latency p50 / p95               | 1,798 ms / 2,472 ms        |
| Retrieval p50 / generation p50        | 744 ms / 1,097 ms          |

Every in-domain question was answered and no answer cited a source that does not exist. Four of five off-topic questions were refused before the LLM was called, because no paper scored above 0.3.

### Known issues

- **Answers from the model's own knowledge.** The poem request (e17) and two out-of-corpus questions (e18, e19) were answered without citations and with a groundedness of 0. The model answered from its training data instead of the retrieved papers. A stricter system prompt and a check that replaces answers without citations with a refusal are planned.
- **Answers without citations.** A third of the answers cited no paper, even when their content matched the sources.
- **Retrieval latency.** Retrieval takes about 700 ms, compared with about 15 ms in the earlier retrieval benchmark. Each search reads a stored vector back from ChromaDB to check its size, and this check should be cached.
- **Chunking** is not implemented yet. Abstracts longer than 256 tokens are cut off when they are embedde

# Parallax-Search

This project collects 5,000 recent AI papers from arXiv, cleans them into a text dataset, embeds them with sentence-transformers and stores them in ChromaDB for semantic search. Questions are answered by an LLM using the papers that the search finds. Each paper is also tagged with a topic and its named entities, so searches can be filtered. Each paper is stored as one document containing its title and abstract.

## Project structure

```
Internship-project-Parallax-Labs/
├── data/
│   ├── raw/                          PDFs, one per paper
│   │   └── rejected/                 papers removed by the language filter
│   ├── processed/
│   │   ├── clean_corpus.jsonl        cleaned dataset
│   │   ├── embeddings.npy            document vectors
│   │   ├── embedding_ids.json        document IDs, same order as the vectors
│   │   ├── doc_topics.jsonl          topic of each paper
│   │   ├── topic_info.csv            list of topics with their top words
│   │   ├── topic_names.json          topic names from the manual review
│   │   └── doc_entities.jsonl        named entities of each paper
│   └── chroma/                       ChromaDB database
├── docs/
│   ├── chunking_and_vector_store.md
│   └── nlp_metadata.md
├── eval/
│   ├── ner_gold.csv                  hand-labeled entities for 50 papers
│   └── topic_review.csv              manual review of each topic
├── logs/
│   ├── embeddings.log
│   ├── retrieval.log
│   ├── retrieval_report.json
│   ├── topics.log
│   ├── topic_metrics.json
│   ├── topic_review_summary.json
│   ├── ner_eval_baseline.json
│   └── ner_eval_with_rules.json
├── models/
│   └── bertopic/                     saved topic model
├── reports/
│   ├── topics_barchart.html          top words of each topic
│   └── topics_map.html               map of how topics relate
├── scripts/
│   ├── verify_env.py
│   ├── fetch_arxiv.py
│   ├── build_corpus.py
│   └── test_retrieval.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── prompts.py
│   ├── rag.py
│   ├── topics.py
│   ├── topic_quality.py
│   └── ner.py
├── tests/
│   └── test_preprocessing.py
├── .env.example
├── requirements.txt
└── README.md
```

The `data/`, `logs/` and `models/` folders, `.env` and `venv/` are in `.gitignore`, so none of them are in the repository. Run the steps below to generate them. The `eval/` folder is committed, because it holds manual labels that cannot be regenerated.

## Setup

Python 3.10 or newer is required. The scripts were written and run on Windows.

Create and activate a virtual environment from the project root:

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
venv\Scripts\activate.bat        # CMD
source venv/bin/activate         # macOS / Linux
```

Install PyTorch first. For a CUDA build, get the install command from https://pytorch.org/get-started/locally/. Otherwise:

```bash
pip install torch
```

Then install the rest:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### API key

Answer generation needs an API key from an LLM provider. The default is Groq, which has a free tier. Create a key at https://console.groq.com/keys, then copy `.env.example` to `.env` and fill it in:

```
LLM_API_KEY=gsk_your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
LLM_REASONING_EFFORT=low
```

`.env` is never committed. `.env.example` lists the same settings without a key.

## Running the pipeline

Run these from the project root with the virtual environment active, in this order.

### 1. Check the environment

```bash
python scripts/verify_env.py
```

Prints the PyTorch version and whether CUDA is available. A CPU-only machine will print "CUDA is not available", which is fine.

### 2. Download the papers

```bash
python scripts/fetch_arxiv.py
```

Uses the arXiv API to get the 5,000 most recent papers from cs.AI, cs.CL and cs.LG. For each paper it creates a one-page PDF with the title and abstract and saves it to `data/raw/`, named after the paper title. Characters that Windows does not allow in file names are removed, and long titles are cut to 150 characters. If two papers have the same title, the arXiv ID is added to the second file name.

This takes a few minutes. Files that already exist are skipped, so the script can be run again if it stops.

### 3. Run the tests

```bash
python -m pytest tests -v
```

Tests the cleaning functions before they are used on the data. All tests should pass before moving on.

### 4. Clean the PDFs

Back up the raw data first, since this step overwrites the files:

```powershell
Copy-Item data\raw data\raw_backup -Recurse
```

Then run:

```bash
python src/preprocessing.py
```

Each PDF in `data/raw/` is read, cleaned and saved again under the same name. Papers that are not in English are moved to `data/raw/rejected/`. A summary of how many files were cleaned, rejected or failed is printed at the end.

### 5. Build the dataset

```bash
python scripts/build_corpus.py
```

Reads the cleaned PDFs, checks each document and writes the final dataset to `data/processed/clean_corpus.jsonl`.

### 6. Generate embeddings

```bash
python src/embeddings.py
```

Embeds the `text` field of every document with `all-MiniLM-L6-v2` and saves the vectors to `data/processed/embeddings.npy` and the matching IDs to `data/processed/embedding_ids.json`. The model is downloaded on the first run (about 90 MB) and cached after that.

To try several batch sizes first and use the fastest one:

```bash
python src/embeddings.py --benchmark
```

### 7. Find topics

```bash
python src/topics.py
```

Groups the papers into topics with BERTopic, reusing the saved embeddings. Writes the topic of each paper to `data/processed/doc_topics.jsonl`, the topic list to `data/processed/topic_info.csv`, the model to `models/bertopic/` and two charts to `reports/`. Takes about a minute on CPU.

Options:

```bash
python src/topics.py --min-cluster-size 10       # allow smaller topics
python src/topics.py --nr-topics 30              # merge similar topics down to 30
python src/topics.py --outlier-threshold 0.6     # stricter outlier reassignment (0 turns it off)
```

### 8. Check topic quality

```bash
python src/topic_quality.py metrics
python src/topic_quality.py review
```

`metrics` prints and saves automatic quality numbers. `review` creates `eval/topic_review.csv` for checking each topic by hand. Fill in the `coherent` column (`yes`, `partly` or `no`) and optionally `better_name` and `notes`, then run:

```bash
python src/topic_quality.py summarize
```

This saves the results to `logs/topic_review_summary.json` and your topic names to `data/processed/topic_names.json`.

### 9. Extract named entities

```bash
python src/ner.py run
```

Finds organizations, models and datasets, people, places and languages in every paper, and writes them to `data/processed/doc_entities.jsonl`. Takes about two minutes on CPU.

To measure accuracy, create a sample of 50 papers to label by hand (once):

```bash
python src/ner.py sample
```

Fill in the `gold_entities` column of `eval/ner_gold.csv` in the form `OpenAI|ORG; GPT-4|PRODUCT`, or `NONE` for a paper without entities. Then compare plain spaCy with the improved version:

```bash
python src/ner.py evaluate --baseline
python src/ner.py evaluate
```

### 10. Load everything into ChromaDB

```bash
python src/vector_store.py ingest --reset
```

Stores every paper's ID, vector, text and metadata, including its topic and entities, in the ChromaDB database in `data/chroma/`. Run it again with `--reset` whenever the embeddings, topics, entities or topic names change. It takes under a minute.

### 11. Search

```bash
python src/vector_store.py search "making large language models safer"
python src/vector_store.py search "graph neural networks for molecules" -k 10 --min-score 0.3
```

Prints the top results with their similarity score, topic, entities and the time the search took. `-k` sets the number of results and `--min-score` hides results below a score.

Searches can be filtered by topic and entity:

```bash
python src/vector_store.py topics
python src/vector_store.py search "reward models for alignment" --topic 0
python src/vector_store.py search "open-weight model evaluation" --entity Qwen3
python src/vector_store.py search "tool use" --topic 0 --no-reassigned
```

`topics` lists the topic numbers. `--no-reassigned` leaves out papers whose topic came from outlier reassignment.

### 12. Test retrieval performance

```bash
python scripts/test_retrieval.py
```

Runs a fixed set of test queries and measures latency and retrieval quality. Results are written to `logs/retrieval.log` and `logs/retrieval_report.json`.

### 13. Check the LLM connection

```bash
python src/llm.py --list-models
python src/llm.py --ping
```

`--list-models` prints the model IDs that are live for your API key and warns if `LLM_MODEL` is not one of them. Free model IDs change often, so run this first whenever a request fails with "model not found". `--ping` sends a short test request and prints the reply, latency and token usage.

### 14. Ask questions

```bash
python src/rag.py "What methods are used to make large language models safer?"
```

Or start interactive mode, which loads the models once and takes one question after another until an empty line is entered:

```bash
python src/rag.py
```

Each answer is followed by the list of source papers with their scores, and by the model name, latency and token usage of the LLM call.

## Text cleaning

The functions are in `src/preprocessing.py` and are applied in this order:

1. `strip_html` removes HTML tags and decodes entities such as `&amp;`. It only parses the text as HTML when it finds a real tag like `<p>` or `<br>`, so math in abstracts such as `x<y` is left alone.
2. `normalize_unicode` applies NFKC normalization, which turns ligatures like `ﬁ` into `fi` and non-breaking spaces into normal spaces. It also removes zero-width and control characters.
3. `clean_whitespace` collapses line breaks, tabs and repeated spaces into single spaces.
4. `detect_language` identifies the language with langdetect. Only English documents are kept.

`clean_text` runs the first three steps together.

## Dataset

The cleaned dataset is at:

```
data/processed/clean_corpus.jsonl
```

It has one JSON object per line, one line per paper:

| Field           | Description                                          |
| --------------- | ---------------------------------------------------- |
| `id`          | 12-character hash of the text, the same on every run |
| `title`       | paper title                                          |
| `abstract`    | paper abstract                                       |
| `text`        | title and abstract together                          |
| `lang`        | detected language, always `en`                     |
| `n_words`     | number of words in `text`                          |
| `source_file` | name of the PDF in `data/raw/`                     |

Example:

```json
{"id": "3f9a1c2b7d4e", "title": "Example Paper Title", "abstract": "We propose a method that...", "text": "Example Paper Title We propose a method that...", "lang": "en", "n_words": 187, "source_file": "Example Paper Title.pdf"}
```

A document is left out of the dataset if it is empty, has fewer than 20 words, is not in English, would still change if cleaned again, or has the same text as a document already added. After writing the file, the script reads it back and checks that every line has the expected fields, that no ID appears twice and that the number of lines is correct.

To load it:

```python
import pandas as pd

df = pd.read_json("data/processed/clean_corpus.jsonl", lines=True)
```

## Embeddings

`src/embeddings.py` uses `all-MiniLM-L6-v2` from sentence-transformers. It produces 384-dimensional vectors, runs well on CPU and reads up to 256 tokens per text. Vectors are normalized to unit length so that similarity can be compared with a simple dot product.

Before timing starts, the model encodes a couple of texts as a warm-up, so one-time setup cost does not affect the numbers. Each run logs the following to the console and to `logs/embeddings.log`:

- model load time and device
- how many texts are longer than 256 tokens and will be truncated
- total time, texts per second and milliseconds per text

| File                                  | Contents                                                 |
| ------------------------------------- | -------------------------------------------------------- |
| `data/processed/embeddings.npy`     | array of shape `(number of documents, 384)`            |
| `data/processed/embedding_ids.json` | document IDs, in the same order as the rows of the array |

## Topics

`src/topics.py` uses BERTopic. It reduces the saved embeddings to 5 dimensions with UMAP, groups them with HDBSCAN, and describes each group with its most distinctive words. BERTopic was chosen over LDA because it groups papers by meaning rather than by word counts, and LDA works poorly on short texts such as abstracts.

Words that appear in almost every abstract, such as "propose", "method" and "results", are removed before topic words are chosen, so they do not end up in every label. The random seed is fixed, so the topics are the same on every run.

Three edge cases are handled:

- **Outliers.** HDBSCAN leaves papers that fit no topic in topic -1. These are moved to their most similar topic if the similarity is at least 0.5, and marked as reassigned.
- **Short documents.** Papers under 60 words are marked, since their topics are less reliable.
- **Jargon.** Words that appear among the top words of 25% or more of the topics are reported as candidates for the stop word list.

First run on 5,007 papers:

| Metric                               | Value                       |
| ------------------------------------ | --------------------------- |
| Topics                               | 42                          |
| Outliers before / after reassignment | 1,622 (32.4%) / 201 (4.0%)  |
| Papers shorter than 60 words         | 0                           |
| Topic diversity                      | 0.729                       |
| Mean cohesion (whole corpus: 0.517)  | 0.666                       |
| Largest topic                        | Topic 0, agents, 759 papers |

`src/topic_quality.py` computes the metrics, creates the manual review sheet and summarizes it. Cohesion is the average similarity between the papers of a topic and its center, and diversity is the share of unique words among the top 10 words of all topics.

## Named entities

`src/ner.py` uses spaCy's `en_core_web_sm` model and keeps five entity types: `ORG`, `PRODUCT` (named models, datasets and benchmarks), `PERSON`, `GPE` (countries and cities) and `LANGUAGE`.

The plain spaCy model was trained mostly on news, so it labeled many AI terms wrongly, for example "LLM" as an organization and "Gaussian" as a person. Three changes were added:

- rules that tag known model families, datasets and companies before spaCy runs
- a list of general and mathematical terms that are never kept as entities
- lowercase spans are dropped, since real names are capitalized

With these changes, `PRODUCT` entities went from 879 to 1,997, and the most frequent ones are now model names such as Qwen3, Claude and DeepSeek.

Sentiment analysis was not used, because research abstracts are written in a neutral tone and almost every paper would be labeled neutral.

## Vector store

`src/vector_store.py` stores the embeddings in a persistent ChromaDB collection named `arxiv_papers` in `data/chroma/`. ChromaDB does not compute any embeddings itself. Both documents and queries are embedded with the same model in our own code, and the collection records which model it was built with, so a search with a different model is refused.

Each record contains the document ID, its vector, its text and this metadata:

| Field                                               | Description                                             |
| --------------------------------------------------- | ------------------------------------------------------- |
| `doc_id`, `title`, `n_words`, `source_file` | from the dataset                                        |
| `topic`, `topic_label`                          | topic number and name                                   |
| `topic_reassigned`                                | true if the topic came from outlier reassignment        |
| `topic_prob`                                      | topic confidence, for papers placed directly in a topic |
| `entities`, `n_entities`                        | readable list of entities and their number              |
| `ent_<name>`                                      | one true/false flag per entity, such as `ent_qwen3`   |

Each entity is stored as its own flag because ChromaDB filters work reliably on simple values in every version. Topic and entity metadata is added during ingest, so `ingest --reset` always rebuilds a clean collection.

The collection uses ChromaDB's default squared L2 distance. Because the vectors are normalized, this gives the same ranking as cosine similarity, and the score shown in search results is the cosine similarity, calculated as `1 - distance / 2`.

Records are written with `upsert` in batches of 1,000, so running ingest twice does not create duplicates. Input is checked before anything is written: missing files, mismatched counts, duplicate or empty IDs, NaN values in vectors, wrong vector sizes and metadata types ChromaDB does not accept are all caught with a clear message. Filtering by a topic that does not exist, or searching a collection that was built before topics and entities were added, also gives a clear message. The full list of edge cases is in `docs/chunking_and_vector_store.md` and `docs/nlp_metadata.md`.

## Retrieval testing

`scripts/test_retrieval.py` loads the model once, runs three warm-up searches and then runs three tests:

1. **Latency by query.** Short, medium, long, non-English and meaningless queries, each run 5 times. Time is split into query encoding and the ChromaDB lookup.
2. **Latency by k.** The same query with 1, 5, 10, 50 and 100 results.
3. **Quality check.** The titles of 200 random papers are used as queries, and the script checks whether each paper comes back. It reports Recall@1, Recall@5 and MRR.

Options:

```bash
python scripts/test_retrieval.py -k 10 --repeats 10 --samples 500
```

Results from the first run on CPU with 5,007 documents:

| Metric                                 | Value      |
| -------------------------------------- | ---------- |
| Median latency                         | 14.61 ms   |
| 95th percentile latency                | 18.21 ms   |
| Share of time spent encoding the query | about 80%  |
| ChromaDB lookup                        | about 2 ms |
| Recall@1                               | 0.985      |
| Recall@5                               | 1.000      |
| MRR                                    | 0.990      |

Relevant queries scored between 0.44 and 0.64, while meaningless queries such as `????` scored between 0.17 and 0.21. A `--min-score` of 0.3 filters out most junk results.

The quality check is a sanity check. The title is part of each document's text, so title searches are expected to find the right paper almost every time. A low score would mean the index is out of date or was built with another model.

## Answer generation

Questions are answered in three steps:

1. **Retrieve.** The question is embedded and the five most similar papers are fetched from ChromaDB. Papers scoring below 0.3 are dropped.
2. **Build the prompt.** The papers are numbered and placed in the prompt as context, together with the question.
3. **Generate.** The LLM writes an answer based only on that context and cites the papers it used as `[1]`, `[2]` and so on.

If no paper scores above 0.3, the LLM is not called and the answer says that no relevant papers were found. This avoids an API call that could only produce an answer without any supporting papers.

### Files

| File               | Purpose                                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/llm.py`     | Client for the LLM API. Reads its settings from `.env`, sends chat requests and returns the answer with latency and token usage. |
| `src/prompts.py` | System prompt and formatting of the retrieved papers into the prompt                                                               |
| `src/rag.py`     | Runs the three steps above. Loads the embedding model, ChromaDB and the LLM client once.                                           |

### Provider and model

The client uses the `openai` Python package, which works with any provider that offers an OpenAI-compatible API. Groq and OpenRouter both do, so switching providers only means changing `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `.env`.

The default model is `openai/gpt-oss-20b` on Groq. It is listed as a production model and is available on the free tier, while several models used in older tutorials are now enterprise-only.

`gpt-oss-20b` is a reasoning model, which spends tokens on reasoning before it answers. `LLM_REASONING_EFFORT=low` keeps that short, which makes answers faster and uses less of the free tier's limit of 8,000 tokens per minute. For a model that does not support reasoning, leave the setting empty.

To use OpenRouter instead:

```
LLM_API_KEY=sk-or-your_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=a-model-id:free
LLM_REASONING_EFFORT=
```

### Settings

| Setting                  | Default                            | Description                                            |
| ------------------------ | ---------------------------------- | ------------------------------------------------------ |
| `LLM_API_KEY`          | none                               | API key for the provider                               |
| `LLM_BASE_URL`         | `https://api.groq.com/openai/v1` | Provider's API address                                 |
| `LLM_MODEL`            | `openai/gpt-oss-20b`             | Model ID                                               |
| `LLM_REASONING_EFFORT` | `low`                            | `low`, `medium`, `high`, or empty to turn it off |
| `LLM_MAX_TOKENS`       | `1024`                           | Maximum length of the answer, including reasoning      |
| `LLM_TEMPERATURE`      | `0.2`                            | Kept low so answers stay close to the context          |

A request for a model ID that does not exist gives a message pointing to `--list-models`, and a rejected API key gives a message pointing to `.env`.

### First results

In the first test, greetings and unrelated inputs such as `hi`, `who are u` and `csc` scored below 0.3, so no LLM call was made. The query `sd` matched one paper at 0.341, and the model asked for clarification instead of guessing. That LLM call took 754 ms and used 437 tokens in tota

# Parallax-Search

This project collects 5,000 recent AI papers from arXiv, cleans them into a text dataset, embeds them with sentence-transformers and stores them in ChromaDB for semantic search. Questions are answered by an LLM using the papers that the search finds. Each paper is stored as one document containing its title and abstract.

## Project structure

```
Internship-project-Parallax-Labs/
├── data/
│   ├── raw/                          PDFs, one per paper
│   │   └── rejected/                 papers removed by the language filter
│   ├── processed/
│   │   ├── clean_corpus.jsonl        cleaned dataset
│   │   ├── embeddings.npy            document vectors
│   │   └── embedding_ids.json        document IDs, same order as the vectors
│   └── chroma/                       ChromaDB database
├── docs/
│   └── chunking_and_vector_store.md
├── logs/
│   ├── embeddings.log
│   ├── retrieval.log
│   └── retrieval_report.json
├── scripts/
│   ├── verify_env.py
│   ├── fetch_arxiv.py
│   ├── build_corpus.py
│   └── test_retrieval.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── prompts.py
│   └── rag.py
├── tests/
│   └── test_preprocessing.py
├── .env.example
├── requirements.txt
└── README.md
```

The `data/` and `logs/` folders, `.env` and `venv/` are in `.gitignore`, so none of them are in the repository. Run the steps below to generate them.

## Setup

Python 3.10 or newer is required. The scripts were written and run on Windows.

Create and activate a virtual environment from the project root:

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
venv\Scripts\activate.bat        # CMD
source venv/bin/activate         # macOS / Linux
```

Install PyTorch first. For a CUDA build, get the install command from https://pytorch.org/get-started/locally/. Otherwise:

```bash
pip install torch
```

Then install the rest:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### API key

Answer generation needs an API key from an LLM provider. The default is Groq, which has a free tier. Create a key at https://console.groq.com/keys, then copy `.env.example` to `.env` and fill it in:

```
LLM_API_KEY=gsk_your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
LLM_REASONING_EFFORT=low
```

`.env` is never committed. `.env.example` lists the same settings without a key.

## Running the pipeline

Run these from the project root with the virtual environment active, in this order.

### 1. Check the environment

```bash
python scripts/verify_env.py
```

Prints the PyTorch version and whether CUDA is available. A CPU-only machine will print "CUDA is not available", which is fine.

### 2. Download the papers

```bash
python scripts/fetch_arxiv.py
```

Uses the arXiv API to get the 5,000 most recent papers from cs.AI, cs.CL and cs.LG. For each paper it creates a one-page PDF with the title and abstract and saves it to `data/raw/`, named after the paper title. Characters that Windows does not allow in file names are removed, and long titles are cut to 150 characters. If two papers have the same title, the arXiv ID is added to the second file name.

This takes a few minutes. Files that already exist are skipped, so the script can be run again if it stops.

### 3. Run the tests

```bash
python -m pytest tests -v
```

Tests the cleaning functions before they are used on the data. All tests should pass before moving on.

### 4. Clean the PDFs

Back up the raw data first, since this step overwrites the files:

```powershell
Copy-Item data\raw data\raw_backup -Recurse
```

Then run:

```bash
python src/preprocessing.py
```

Each PDF in `data/raw/` is read, cleaned and saved again under the same name. Papers that are not in English are moved to `data/raw/rejected/`. A summary of how many files were cleaned, rejected or failed is printed at the end.

### 5. Build the dataset

```bash
python scripts/build_corpus.py
```

Reads the cleaned PDFs, checks each document and writes the final dataset to `data/processed/clean_corpus.jsonl`.

### 6. Generate embeddings

```bash
python src/embeddings.py
```

Embeds the `text` field of every document with `all-MiniLM-L6-v2` and saves the vectors to `data/processed/embeddings.npy` and the matching IDs to `data/processed/embedding_ids.json`. The model is downloaded on the first run (about 90 MB) and cached after that.

To try several batch sizes first and use the fastest one:

```bash
python src/embeddings.py --benchmark
```

### 7. Load the embeddings into ChromaDB

```bash
python src/vector_store.py ingest
```

Stores every document's ID, vector, text and metadata in the ChromaDB database in `data/chroma/`. This only needs to be done once, or again after the embeddings change. To delete the collection and rebuild it from scratch:

```bash
python src/vector_store.py ingest --reset
```

### 8. Search

```bash
python src/vector_store.py search "making large language models safer"
python src/vector_store.py search "graph neural networks for molecules" -k 10 --min-score 0.3
```

Prints the top results with their similarity score and the time the search took. `-k` sets the number of results and `--min-score` hides results below a score.

### 9. Test retrieval performance

```bash
python scripts/test_retrieval.py
```

Runs a fixed set of test queries and measures latency and retrieval quality. Results are written to `logs/retrieval.log` and `logs/retrieval_report.json`.

### 10. Check the LLM connection

```bash
python src/llm.py --list-models
python src/llm.py --ping
```

`--list-models` prints the model IDs that are live for your API key and warns if `LLM_MODEL` is not one of them. Free model IDs change often, so run this first whenever a request fails with "model not found". `--ping` sends a short test request and prints the reply, latency and token usage.

### 11. Ask questions

```bash
python src/rag.py "What methods are used to make large language models safer?"
```

Or start interactive mode, which loads the models once and takes one question after another until an empty line is entered:

```bash
python src/rag.py
```

Each answer is followed by the list of source papers with their scores, and by the model name, latency and token usage of the LLM call.

## Text cleaning

The functions are in `src/preprocessing.py` and are applied in this order:

1. `strip_html` removes HTML tags and decodes entities such as `&amp;`. It only parses the text as HTML when it finds a real tag like `<p>` or `<br>`, so math in abstracts such as `x<y` is left alone.
2. `normalize_unicode` applies NFKC normalization, which turns ligatures like `ﬁ` into `fi` and non-breaking spaces into normal spaces. It also removes zero-width and control characters.
3. `clean_whitespace` collapses line breaks, tabs and repeated spaces into single spaces.
4. `detect_language` identifies the language with langdetect. Only English documents are kept.

`clean_text` runs the first three steps together.

## Dataset

The cleaned dataset is at:

```
data/processed/clean_corpus.jsonl
```

It has one JSON object per line, one line per paper:

| Field           | Description                                          |
| --------------- | ---------------------------------------------------- |
| `id`          | 12-character hash of the text, the same on every run |
| `title`       | paper title                                          |
| `abstract`    | paper abstract                                       |
| `text`        | title and abstract together                          |
| `lang`        | detected language, always `en`                     |
| `n_words`     | number of words in `text`                          |
| `source_file` | name of the PDF in `data/raw/`                     |

Example:

```json
{"id": "3f9a1c2b7d4e", "title": "Example Paper Title", "abstract": "We propose a method that...", "text": "Example Paper Title We propose a method that...", "lang": "en", "n_words": 187, "source_file": "Example Paper Title.pdf"}
```

A document is left out of the dataset if it is empty, has fewer than 20 words, is not in English, would still change if cleaned again, or has the same text as a document already added. After writing the file, the script reads it back and checks that every line has the expected fields, that no ID appears twice and that the number of lines is correct.

To load it:

```python
import pandas as pd

df = pd.read_json("data/processed/clean_corpus.jsonl", lines=True)
```

## Embeddings

`src/embeddings.py` uses `all-MiniLM-L6-v2` from sentence-transformers. It produces 384-dimensional vectors, runs well on CPU and reads up to 256 tokens per text. Vectors are normalized to unit length so that similarity can be compared with a simple dot product.

Before timing starts, the model encodes a couple of texts as a warm-up, so one-time setup cost does not affect the numbers. Each run logs the following to the console and to `logs/embeddings.log`:

- model load time and device
- how many texts are longer than 256 tokens and will be truncated
- total time, texts per second and milliseconds per text

| File                                  | Contents                                                 |
| ------------------------------------- | -------------------------------------------------------- |
| `data/processed/embeddings.npy`     | array of shape `(number of documents, 384)`            |
| `data/processed/embedding_ids.json` | document IDs, in the same order as the rows of the array |

## Vector store

`src/vector_store.py` stores the embeddings in a persistent ChromaDB collection named `arxiv_papers` in `data/chroma/`. ChromaDB does not compute any embeddings itself. Both documents and queries are embedded with the same model in our own code, and the collection records which model it was built with, so a search with a different model is refused.

Each record contains the document ID, its vector, its text and this metadata: `doc_id`, `title`, `n_words` and `source_file`.

The collection uses ChromaDB's default squared L2 distance. Because the vectors are normalized, this gives the same ranking as cosine similarity, and the score shown in search results is the cosine similarity, calculated as `1 - distance / 2`.

Records are written with `upsert` in batches of 1,000, so running ingest twice does not create duplicates. Input is checked before anything is written: missing files, mismatched counts, duplicate or empty IDs, NaN values in vectors, wrong vector sizes and metadata types ChromaDB does not accept are all caught with a clear message. The full list of edge cases and how each one is handled is in `docs/chunking_and_vector_store.md`.

## Retrieval testing

`scripts/test_retrieval.py` loads the model once, runs three warm-up searches and then runs three tests:

1. **Latency by query.** Short, medium, long, non-English and meaningless queries, each run 5 times. Time is split into query encoding and the ChromaDB lookup.
2. **Latency by k.** The same query with 1, 5, 10, 50 and 100 results.
3. **Quality check.** The titles of 200 random papers are used as queries, and the script checks whether each paper comes back. It reports Recall@1, Recall@5 and MRR.

Options:

```bash
python scripts/test_retrieval.py -k 10 --repeats 10 --samples 500
```

Results from the first run on CPU with 5,007 documents:

| Metric                                 | Value      |
| -------------------------------------- | ---------- |
| Median latency                         | 14.61 ms   |
| 95th percentile latency                | 18.21 ms   |
| Share of time spent encoding the query | about 80%  |
| ChromaDB lookup                        | about 2 ms |
| Recall@1                               | 0.985      |
| Recall@5                               | 1.000      |
| MRR                                    | 0.990      |

Relevant queries scored between 0.44 and 0.64, while meaningless queries such as `????` scored between 0.17 and 0.21. A `--min-score` of 0.3 filters out most junk results.

The quality check is a sanity check. The title is part of each document's text, so title searches are expected to find the right paper almost every time. A low score would mean the index is out of date or was built with another model.

## Answer generation

Questions are answered in three steps:

1. **Retrieve.** The question is embedded and the five most similar papers are fetched from ChromaDB. Papers scoring below 0.3 are dropped.
2. **Build the prompt.** The papers are numbered and placed in the prompt as context, together with the question.
3. **Generate.** The LLM writes an answer based only on that context and cites the papers it used as `[1]`, `[2]` and so on.

If no paper scores above 0.3, the LLM is not called and the answer says that no relevant papers were found. This avoids an API call that could only produce an answer without any supporting papers.

### Files

| File               | Purpose                                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/llm.py`     | Client for the LLM API. Reads its settings from `.env`, sends chat requests and returns the answer with latency and token usage. |
| `src/prompts.py` | System prompt and formatting of the retrieved papers into the prompt                                                               |
| `src/rag.py`     | Runs the three steps above. Loads the embedding model, ChromaDB and the LLM client once.                                           |

### Provider and model

The client uses the `openai` Python package, which works with any provider that offers an OpenAI-compatible API. Groq and OpenRouter both do, so switching providers only means changing `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `.env`.

The default model is `openai/gpt-oss-20b` on Groq. It is listed as a production model and is available on the free tier, while several models used in older tutorials are now enterprise-only.

`gpt-oss-20b` is a reasoning model, which spends tokens on reasoning before it answers. `LLM_REASONING_EFFORT=low` keeps that short, which makes answers faster and uses less of the free tier's limit of 8,000 tokens per minute. For a model that does not support reasoning, leave the setting empty.

To use OpenRouter instead:

```
LLM_API_KEY=sk-or-your_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=a-model-id:free
LLM_REASONING_EFFORT=
```

### Settings

| Setting                  | Default                            | Description                                            |
| ------------------------ | ---------------------------------- | ------------------------------------------------------ |
| `LLM_API_KEY`          | none                               | API key for the provider                               |
| `LLM_BASE_URL`         | `https://api.groq.com/openai/v1` | Provider's API address                                 |
| `LLM_MODEL`            | `openai/gpt-oss-20b`             | Model ID                                               |
| `LLM_REASONING_EFFORT` | `low`                            | `low`, `medium`, `high`, or empty to turn it off |
| `LLM_MAX_TOKENS`       | `1024`                           | Maximum length of the answer, including reasoning      |
| `LLM_TEMPERATURE`      | `0.2`                            | Kept low so answers stay close to the context          |

A request for a model ID that does not exist gives a message pointing to `--list-models`, and a rejected API key gives a message pointing to `.env`.

### First results

In the first test, greetings and unrelated inputs such as `hi`, `who are u` and `csc` scored below 0.3, so no LLM call was made. The query `sd` matched one paper at 0.341, and the model asked for clarification instead of guessing. That LLM call took 754 ms and used 437 tokens in tota

# Parallax-Search

This project collects 5,000 recent AI papers from arXiv, cleans them into a text dataset, embeds them with sentence-transformers and stores them in ChromaDB for semantic search. Questions are answered by an LLM using the papers that the search finds. Each paper is stored as one document containing its title and abstract.

## Project structure

```
Internship-project-Parallax-Labs/
├── data/
│   ├── raw/                          PDFs, one per paper
│   │   └── rejected/                 papers removed by the language filter
│   ├── processed/
│   │   ├── clean_corpus.jsonl        cleaned dataset
│   │   ├── embeddings.npy            document vectors
│   │   └── embedding_ids.json        document IDs, same order as the vectors
│   └── chroma/                       ChromaDB database
├── docs/
│   └── chunking_and_vector_store.md
├── logs/
│   ├── embeddings.log
│   ├── retrieval.log
│   └── retrieval_report.json
├── scripts/
│   ├── verify_env.py
│   ├── fetch_arxiv.py
│   ├── build_corpus.py
│   └── test_retrieval.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── prompts.py
│   └── rag.py
├── tests/
│   └── test_preprocessing.py
├── .env.example
├── requirements.txt
└── README.md
```

The `data/` and `logs/` folders, `.env` and `venv/` are in `.gitignore`, so none of them are in the repository. Run the steps below to generate them.

## Setup

Python 3.10 or newer is required. The scripts were written and run on Windows.

Create and activate a virtual environment from the project root:

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
venv\Scripts\activate.bat        # CMD
source venv/bin/activate         # macOS / Linux
```

Install PyTorch first. For a CUDA build, get the install command from https://pytorch.org/get-started/locally/. Otherwise:

```bash
pip install torch
```

Then install the rest:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### API key

Answer generation needs an API key from an LLM provider. The default is Groq, which has a free tier. Create a key at https://console.groq.com/keys, then copy `.env.example` to `.env` and fill it in:

```
LLM_API_KEY=gsk_your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
LLM_REASONING_EFFORT=low
```

`.env` is never committed. `.env.example` lists the same settings without a key.

## Running the pipeline

Run these from the project root with the virtual environment active, in this order.

### 1. Check the environment

```bash
python scripts/verify_env.py
```

Prints the PyTorch version and whether CUDA is available. A CPU-only machine will print "CUDA is not available", which is fine.

### 2. Download the papers

```bash
python scripts/fetch_arxiv.py
```

Uses the arXiv API to get the 5,000 most recent papers from cs.AI, cs.CL and cs.LG. For each paper it creates a one-page PDF with the title and abstract and saves it to `data/raw/`, named after the paper title. Characters that Windows does not allow in file names are removed, and long titles are cut to 150 characters. If two papers have the same title, the arXiv ID is added to the second file name.

This takes a few minutes. Files that already exist are skipped, so the script can be run again if it stops.

### 3. Run the tests

```bash
python -m pytest tests -v
```

Tests the cleaning functions before they are used on the data. All tests should pass before moving on.

### 4. Clean the PDFs

Back up the raw data first, since this step overwrites the files:

```powershell
Copy-Item data\raw data\raw_backup -Recurse
```

Then run:

```bash
python src/preprocessing.py
```

Each PDF in `data/raw/` is read, cleaned and saved again under the same name. Papers that are not in English are moved to `data/raw/rejected/`. A summary of how many files were cleaned, rejected or failed is printed at the end.

### 5. Build the dataset

```bash
python scripts/build_corpus.py
```

Reads the cleaned PDFs, checks each document and writes the final dataset to `data/processed/clean_corpus.jsonl`.

### 6. Generate embeddings

```bash
python src/embeddings.py
```

Embeds the `text` field of every document with `all-MiniLM-L6-v2` and saves the vectors to `data/processed/embeddings.npy` and the matching IDs to `data/processed/embedding_ids.json`. The model is downloaded on the first run (about 90 MB) and cached after that.

To try several batch sizes first and use the fastest one:

```bash
python src/embeddings.py --benchmark
```

### 7. Load the embeddings into ChromaDB

```bash
python src/vector_store.py ingest
```

Stores every document's ID, vector, text and metadata in the ChromaDB database in `data/chroma/`. This only needs to be done once, or again after the embeddings change. To delete the collection and rebuild it from scratch:

```bash
python src/vector_store.py ingest --reset
```

### 8. Search

```bash
python src/vector_store.py search "making large language models safer"
python src/vector_store.py search "graph neural networks for molecules" -k 10 --min-score 0.3
```

Prints the top results with their similarity score and the time the search took. `-k` sets the number of results and `--min-score` hides results below a score.

### 9. Test retrieval performance

```bash
python scripts/test_retrieval.py
```

Runs a fixed set of test queries and measures latency and retrieval quality. Results are written to `logs/retrieval.log` and `logs/retrieval_report.json`.

### 10. Check the LLM connection

```bash
python src/llm.py --list-models
python src/llm.py --ping
```

`--list-models` prints the model IDs that are live for your API key and warns if `LLM_MODEL` is not one of them. Free model IDs change often, so run this first whenever a request fails with "model not found". `--ping` sends a short test request and prints the reply, latency and token usage.

### 11. Ask questions

```bash
python src/rag.py "What methods are used to make large language models safer?"
```

Or start interactive mode, which loads the models once and takes one question after another until an empty line is entered:

```bash
python src/rag.py
```

Each answer is followed by the list of source papers with their scores, and by the model name, latency and token usage of the LLM call.

## Text cleaning

The functions are in `src/preprocessing.py` and are applied in this order:

1. `strip_html` removes HTML tags and decodes entities such as `&amp;`. It only parses the text as HTML when it finds a real tag like `<p>` or `<br>`, so math in abstracts such as `x<y` is left alone.
2. `normalize_unicode` applies NFKC normalization, which turns ligatures like `ﬁ` into `fi` and non-breaking spaces into normal spaces. It also removes zero-width and control characters.
3. `clean_whitespace` collapses line breaks, tabs and repeated spaces into single spaces.
4. `detect_language` identifies the language with langdetect. Only English documents are kept.

`clean_text` runs the first three steps together.

## Dataset

The cleaned dataset is at:

```
data/processed/clean_corpus.jsonl
```

It has one JSON object per line, one line per paper:

| Field           | Description                                          |
| --------------- | ---------------------------------------------------- |
| `id`          | 12-character hash of the text, the same on every run |
| `title`       | paper title                                          |
| `abstract`    | paper abstract                                       |
| `text`        | title and abstract together                          |
| `lang`        | detected language, always `en`                     |
| `n_words`     | number of words in `text`                          |
| `source_file` | name of the PDF in `data/raw/`                     |

Example:

```json
{"id": "3f9a1c2b7d4e", "title": "Example Paper Title", "abstract": "We propose a method that...", "text": "Example Paper Title We propose a method that...", "lang": "en", "n_words": 187, "source_file": "Example Paper Title.pdf"}
```

A document is left out of the dataset if it is empty, has fewer than 20 words, is not in English, would still change if cleaned again, or has the same text as a document already added. After writing the file, the script reads it back and checks that every line has the expected fields, that no ID appears twice and that the number of lines is correct.

To load it:

```python
import pandas as pd

df = pd.read_json("data/processed/clean_corpus.jsonl", lines=True)
```

## Embeddings

`src/embeddings.py` uses `all-MiniLM-L6-v2` from sentence-transformers. It produces 384-dimensional vectors, runs well on CPU and reads up to 256 tokens per text. Vectors are normalized to unit length so that similarity can be compared with a simple dot product.

Before timing starts, the model encodes a couple of texts as a warm-up, so one-time setup cost does not affect the numbers. Each run logs the following to the console and to `logs/embeddings.log`:

- model load time and device
- how many texts are longer than 256 tokens and will be truncated
- total time, texts per second and milliseconds per text

| File                                  | Contents                                                 |
| ------------------------------------- | -------------------------------------------------------- |
| `data/processed/embeddings.npy`     | array of shape `(number of documents, 384)`            |
| `data/processed/embedding_ids.json` | document IDs, in the same order as the rows of the array |

## Vector store

`src/vector_store.py` stores the embeddings in a persistent ChromaDB collection named `arxiv_papers` in `data/chroma/`. ChromaDB does not compute any embeddings itself. Both documents and queries are embedded with the same model in our own code, and the collection records which model it was built with, so a search with a different model is refused.

Each record contains the document ID, its vector, its text and this metadata: `doc_id`, `title`, `n_words` and `source_file`.

The collection uses ChromaDB's default squared L2 distance. Because the vectors are normalized, this gives the same ranking as cosine similarity, and the score shown in search results is the cosine similarity, calculated as `1 - distance / 2`.

Records are written with `upsert` in batches of 1,000, so running ingest twice does not create duplicates. Input is checked before anything is written: missing files, mismatched counts, duplicate or empty IDs, NaN values in vectors, wrong vector sizes and metadata types ChromaDB does not accept are all caught with a clear message. The full list of edge cases and how each one is handled is in `docs/chunking_and_vector_store.md`.

## Retrieval testing

`scripts/test_retrieval.py` loads the model once, runs three warm-up searches and then runs three tests:

1. **Latency by query.** Short, medium, long, non-English and meaningless queries, each run 5 times. Time is split into query encoding and the ChromaDB lookup.
2. **Latency by k.** The same query with 1, 5, 10, 50 and 100 results.
3. **Quality check.** The titles of 200 random papers are used as queries, and the script checks whether each paper comes back. It reports Recall@1, Recall@5 and MRR.

Options:

```bash
python scripts/test_retrieval.py -k 10 --repeats 10 --samples 500
```

Results from the first run on CPU with 5,007 documents:

| Metric                                 | Value      |
| -------------------------------------- | ---------- |
| Median latency                         | 14.61 ms   |
| 95th percentile latency                | 18.21 ms   |
| Share of time spent encoding the query | about 80%  |
| ChromaDB lookup                        | about 2 ms |
| Recall@1                               | 0.985      |
| Recall@5                               | 1.000      |
| MRR                                    | 0.990      |

Relevant queries scored between 0.44 and 0.64, while meaningless queries such as `????` scored between 0.17 and 0.21. A `--min-score` of 0.3 filters out most junk results.

The quality check is a sanity check. The title is part of each document's text, so title searches are expected to find the right paper almost every time. A low score would mean the index is out of date or was built with another model.

## Answer generation

Questions are answered in three steps:

1. **Retrieve.** The question is embedded and the five most similar papers are fetched from ChromaDB. Papers scoring below 0.3 are dropped.
2. **Build the prompt.** The papers are numbered and placed in the prompt as context, together with the question.
3. **Generate.** The LLM writes an answer based only on that context and cites the papers it used as `[1]`, `[2]` and so on.

If no paper scores above 0.3, the LLM is not called and the answer says that no relevant papers were found. This avoids an API call that could only produce an answer without any supporting papers.

### Files

| File               | Purpose                                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/llm.py`     | Client for the LLM API. Reads its settings from `.env`, sends chat requests and returns the answer with latency and token usage. |
| `src/prompts.py` | System prompt and formatting of the retrieved papers into the prompt                                                               |
| `src/rag.py`     | Runs the three steps above. Loads the embedding model, ChromaDB and the LLM client once.                                           |

### Provider and model

The client uses the `openai` Python package, which works with any provider that offers an OpenAI-compatible API. Groq and OpenRouter both do, so switching providers only means changing `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `.env`.

The default model is `openai/gpt-oss-20b` on Groq. It is listed as a production model and is available on the free tier, while several models used in older tutorials are now enterprise-only.

`gpt-oss-20b` is a reasoning model, which spends tokens on reasoning before it answers. `LLM_REASONING_EFFORT=low` keeps that short, which makes answers faster and uses less of the free tier's limit of 8,000 tokens per minute. For a model that does not support reasoning, leave the setting empty.

To use OpenRouter instead:

```
LLM_API_KEY=sk-or-your_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=a-model-id:free
LLM_REASONING_EFFORT=
```

### Settings

| Setting                  | Default                            | Description                                            |
| ------------------------ | ---------------------------------- | ------------------------------------------------------ |
| `LLM_API_KEY`          | none                               | API key for the provider                               |
| `LLM_BASE_URL`         | `https://api.groq.com/openai/v1` | Provider's API address                                 |
| `LLM_MODEL`            | `openai/gpt-oss-20b`             | Model ID                                               |
| `LLM_REASONING_EFFORT` | `low`                            | `low`, `medium`, `high`, or empty to turn it off |
| `LLM_MAX_TOKENS`       | `1024`                           | Maximum length of the answer, including reasoning      |
| `LLM_TEMPERATURE`      | `0.2`                            | Kept low so answers stay close to the context          |

A request for a model ID that does not exist gives a message pointing to `--list-models`, and a rejected API key gives a message pointing to `.env`.

### First results

In the first test, greetings and unrelated inputs such as `hi`, `who are u` and `csc` scored below 0.3, so no LLM call was made. The query `sd` matched one paper at 0.341, and the model asked for clarification instead of guessing. That LLM call took 754 ms and used 437 tokens in tota

# Parallax-Search

This project collects 5,000 recent AI papers from arXiv, cleans them into a text dataset, embeds them with sentence-transformers and stores them in ChromaDB for semantic search. Each paper is stored as one document containing its title and abstract.

## Project structure

```
Internship-project-Parallax-Labs/
├── data/
│   ├── raw/                          PDFs, one per paper
│   │   └── rejected/                 papers removed by the language filter
│   ├── processed/
│   │   ├── clean_corpus.jsonl        cleaned dataset
│   │   ├── embeddings.npy            document vectors
│   │   └── embedding_ids.json        document IDs, same order as the vectors
│   └── chroma/                       ChromaDB database
├── docs/
│   └── chunking_and_vector_store.md
├── logs/
│   ├── embeddings.log
│   ├── retrieval.log
│   └── retrieval_report.json
├── scripts/
│   ├── verify_env.py
│   ├── fetch_arxiv.py
│   ├── build_corpus.py
│   └── test_retrieval.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   └── vector_store.py
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

The `data/` and `logs/` folders, `.env` and `venv/` are in `.gitignore`, so none of them are in the repository. Run the steps below to generate them.

## Setup

Python 3.10 or newer is required. The scripts were written and run on Windows.

Create and activate a virtual environment from the project root:

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
venv\Scripts\activate.bat        # CMD
source venv/bin/activate         # macOS / Linux
```

Install PyTorch first. For a CUDA build, get the install command from https://pytorch.org/get-started/locally/. Otherwise:

```bash
pip install torch
```

Then install the rest:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the pipeline

Run these from the project root with the virtual environment active, in this order.

### 1. Check the environment

```bash
python scripts/verify_env.py
```

Prints the PyTorch version and whether CUDA is available. A CPU-only machine will print "CUDA is not available", which is fine.

### 2. Download the papers

```bash
python scripts/fetch_arxiv.py
```

Uses the arXiv API to get the 5,000 most recent papers from cs.AI, cs.CL and cs.LG. For each paper it creates a one-page PDF with the title and abstract and saves it to `data/raw/`, named after the paper title. Characters that Windows does not allow in file names are removed, and long titles are cut to 150 characters. If two papers have the same title, the arXiv ID is added to the second file name.

This takes a few minutes. Files that already exist are skipped, so the script can be run again if it stops.

### 3. Run the tests

```bash
python -m pytest tests -v
```

Tests the cleaning functions before they are used on the data. All tests should pass before moving on.

### 4. Clean the PDFs

Back up the raw data first, since this step overwrites the files:

```powershell
Copy-Item data\raw data\raw_backup -Recurse
```

Then run:

```bash
python src/preprocessing.py
```

Each PDF in `data/raw/` is read, cleaned and saved again under the same name. Papers that are not in English are moved to `data/raw/rejected/`. A summary of how many files were cleaned, rejected or failed is printed at the end.

### 5. Build the dataset

```bash
python scripts/build_corpus.py
```

Reads the cleaned PDFs, checks each document and writes the final dataset to `data/processed/clean_corpus.jsonl`.

### 6. Generate embeddings

```bash
python src/embeddings.py
```

Embeds the `text` field of every document with `all-MiniLM-L6-v2` and saves the vectors to `data/processed/embeddings.npy` and the matching IDs to `data/processed/embedding_ids.json`. The model is downloaded on the first run (about 90 MB) and cached after that.

To try several batch sizes first and use the fastest one:

```bash
python src/embeddings.py --benchmark
```

### 7. Load the embeddings into ChromaDB

```bash
python src/vector_store.py ingest
```

Stores every document's ID, vector, text and metadata in the ChromaDB database in `data/chroma/`. This only needs to be done once, or again after the embeddings change. To delete the collection and rebuild it from scratch:

```bash
python src/vector_store.py ingest --reset
```

### 8. Search

```bash
python src/vector_store.py search "making large language models safer"
python src/vector_store.py search "graph neural networks for molecules" -k 10 --min-score 0.3
```

Prints the top results with their similarity score and the time the search took. `-k` sets the number of results and `--min-score` hides results below a score.

### 9. Test retrieval performance

```bash
python scripts/test_retrieval.py
```

Runs a fixed set of test queries and measures latency and retrieval quality. Results are written to `logs/retrieval.log` and `logs/retrieval_report.json`.

## Text cleaning

The functions are in `src/preprocessing.py` and are applied in this order:

1. `strip_html` removes HTML tags and decodes entities such as `&amp;`. It only parses the text as HTML when it finds a real tag like `<p>` or `<br>`, so math in abstracts such as `x<y` is left alone.
2. `normalize_unicode` applies NFKC normalization, which turns ligatures like `ﬁ` into `fi` and non-breaking spaces into normal spaces. It also removes zero-width and control characters.
3. `clean_whitespace` collapses line breaks, tabs and repeated spaces into single spaces.
4. `detect_language` identifies the language with langdetect. Only English documents are kept.

`clean_text` runs the first three steps together.

## Dataset

The cleaned dataset is at:

```
data/processed/clean_corpus.jsonl
```

It has one JSON object per line, one line per paper:

| Field           | Description                                          |
| --------------- | ---------------------------------------------------- |
| `id`          | 12-character hash of the text, the same on every run |
| `title`       | paper title                                          |
| `abstract`    | paper abstract                                       |
| `text`        | title and abstract together                          |
| `lang`        | detected language, always `en`                     |
| `n_words`     | number of words in `text`                          |
| `source_file` | name of the PDF in `data/raw/`                     |

Example:

```json
{"id": "3f9a1c2b7d4e", "title": "Example Paper Title", "abstract": "We propose a method that...", "text": "Example Paper Title We propose a method that...", "lang": "en", "n_words": 187, "source_file": "Example Paper Title.pdf"}
```

A document is left out of the dataset if it is empty, has fewer than 20 words, is not in English, would still change if cleaned again, or has the same text as a document already added. After writing the file, the script reads it back and checks that every line has the expected fields, that no ID appears twice and that the number of lines is correct.

To load it:

```python
import pandas as pd

df = pd.read_json("data/processed/clean_corpus.jsonl", lines=True)
```

## Embeddings

`src/embeddings.py` uses `all-MiniLM-L6-v2` from sentence-transformers. It produces 384-dimensional vectors, runs well on CPU and reads up to 256 tokens per text. Vectors are normalized to unit length so that similarity can be compared with a simple dot product.

Before timing starts, the model encodes a couple of texts as a warm-up, so one-time setup cost does not affect the numbers. Each run logs the following to the console and to `logs/embeddings.log`:

- model load time and device
- how many texts are longer than 256 tokens and will be truncated
- total time, texts per second and milliseconds per text

| File                                  | Contents                                                 |
| ------------------------------------- | -------------------------------------------------------- |
| `data/processed/embeddings.npy`     | array of shape `(number of documents, 384)`            |
| `data/processed/embedding_ids.json` | document IDs, in the same order as the rows of the array |

## Vector store

`src/vector_store.py` stores the embeddings in a persistent ChromaDB collection named `arxiv_papers` in `data/chroma/`. ChromaDB does not compute any embeddings itself. Both documents and queries are embedded with the same model in our own code, and the collection records which model it was built with, so a search with a different model is refused.

Each record contains the document ID, its vector, its text and this metadata: `doc_id`, `title`, `n_words` and `source_file`.

The collection uses ChromaDB's default squared L2 distance. Because the vectors are normalized, this gives the same ranking as cosine similarity, and the score shown in search results is the cosine similarity, calculated as `1 - distance / 2`.

Records are written with `upsert` in batches of 1,000, so running ingest twice does not create duplicates. Input is checked before anything is written: missing files, mismatched counts, duplicate or empty IDs, NaN values in vectors, wrong vector sizes and metadata types ChromaDB does not accept are all caught with a clear message. The full list of edge cases and how each one is handled is in `docs/chunking_and_vector_store.md`.

## Retrieval testing

`scripts/test_retrieval.py` loads the model once, runs three warm-up searches and then runs three tests:

1. **Latency by query.** Short, medium, long, non-English and meaningless queries, each run 5 times. Time is split into query encoding and the ChromaDB lookup.
2. **Latency by k.** The same query with 1, 5, 10, 50 and 100 results.
3. **Quality check.** The titles of 200 random papers are used as queries, and the script checks whether each paper comes back. It reports Recall@1, Recall@5 and MRR.

Options:

```bash
python scripts/test_retrieval.py -k 10 --repeats 10 --samples 500
```

Results from the first run on CPU with 5,007 documents:

| Metric                                 | Value      |
| -------------------------------------- | ---------- |
| Median latency                         | 14.61 ms   |
| 95th percentile latency                | 18.21 ms   |
| Share of time spent encoding the query | about 80%  |
| ChromaDB lookup                        | about 2 ms |
| Recall@1                               | 0.985      |
| Recall@5                               | 1.000      |
| MRR                                    | 0.990      |

Relevant queries scored between 0.44 and 0.64, while meaningless queries such as `????` scored between 0.17 and 0.21. A `--min-score` of 0.3 filters out most junk results.

The quality check is a sanity check. The title is part of each document's text, so title searches are expected to find the right paper almost every time. A low score would mean the index is out of date or was built with another mode

# Parallax-Search

This project collects 5,000 recent AI papers from arXiv and turns them into a clean text dataset. Each paper is stored as one document containing its title and abstract.

## Project structure

```
Internship-project-Parallax-Labs/
├── data/
│   ├── raw/                  PDFs, one per paper
│   │   └── rejected/         papers removed by the language filter
│   └── processed/
│       └── clean_corpus.jsonl
├── scripts/
│   ├── verify_env.py
│   ├── fetch_arxiv.py
│   └── build_corpus.py
├── src/
│   ├── __init__.py
│   └── preprocessing.py
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

The `data/` folder, `.env` and `venv/` are in `.gitignore`, so the data is not in the repository. Run the steps below to generate it.

## Setup

Python 3.10 or newer is required. The scripts were written and run on Windows.

Create and activate a virtual environment from the project root:

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
venv\Scripts\activate.bat        # CMD
source venv/bin/activate         # macOS / Linux
```

Install PyTorch first. For a CUDA build, get the install command from https://pytorch.org/get-started/locally/. Otherwise:

```bash
pip install torch
```

Then install the rest:

```bash
# Parallax-Search

This project collects 5,000 recent AI papers from arXiv, cleans them into a text dataset, embeds them with sentence-transformers and stores them in ChromaDB for semantic search. Questions are answered by an LLM using the papers that the search finds. Each paper is stored as one document containing its title and abstract.

## Project structure

```

Internship-project-Parallax-Labs/
├── data/
│   ├── raw/                          PDFs, one per paper
│   │   └── rejected/                 papers removed by the language filter
│   ├── processed/
│   │   ├── clean_corpus.jsonl        cleaned dataset
│   │   ├── embeddings.npy            document vectors
│   │   └── embedding_ids.json        document IDs, same order as the vectors
│   └── chroma/                       ChromaDB database
├── docs/
│   └── chunking_and_vector_store.md
├── logs/
│   ├── embeddings.log
│   ├── retrieval.log
│   └── retrieval_report.json
├── scripts/
│   ├── verify_env.py
│   ├── fetch_arxiv.py
│   ├── build_corpus.py
│   └── test_retrieval.py
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── prompts.py
│   └── rag.py
├── tests/
│   └── test_preprocessing.py
├── .env.example
├── requirements.txt
└── README.md

```

The `data/` and `logs/` folders, `.env` and `venv/` are in `.gitignore`, so none of them are in the repository. Run the steps below to generate them.

## Setup

Python 3.10 or newer is required. The scripts were written and run on Windows.

Create and activate a virtual environment from the project root:

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
venv\Scripts\activate.bat        # CMD
source venv/bin/activate         # macOS / Linux
```

Install PyTorch first. For a CUDA build, get the install command from https://pytorch.org/get-started/locally/. Otherwise:

```bash
pip install torch
```

Then install the rest:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### API key

Answer generation needs an API key from an LLM provider. The default is Groq, which has a free tier. Create a key at https://console.groq.com/keys, then copy `.env.example` to `.env` and fill it in:

```
LLM_API_KEY=gsk_your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
LLM_REASONING_EFFORT=low
```

`.env` is never committed. `.env.example` lists the same settings without a key.

## Running the pipeline

Run these from the project root with the virtual environment active, in this order.

### 1. Check the environment

```bash
python scripts/verify_env.py
```

Prints the PyTorch version and whether CUDA is available. A CPU-only machine will print "CUDA is not available", which is fine.

### 2. Download the papers

```bash
python scripts/fetch_arxiv.py
```

Uses the arXiv API to get the 5,000 most recent papers from cs.AI, cs.CL and cs.LG. For each paper it creates a one-page PDF with the title and abstract and saves it to `data/raw/`, named after the paper title. Characters that Windows does not allow in file names are removed, and long titles are cut to 150 characters. If two papers have the same title, the arXiv ID is added to the second file name.

This takes a few minutes. Files that already exist are skipped, so the script can be run again if it stops.

### 3. Run the tests

```bash
python -m pytest tests -v
```

Tests the cleaning functions before they are used on the data. All tests should pass before moving on.

### 4. Clean the PDFs

Back up the raw data first, since this step overwrites the files:

```powershell
Copy-Item data\raw data\raw_backup -Recurse
```

Then run:

```bash
python src/preprocessing.py
```

Each PDF in `data/raw/` is read, cleaned and saved again under the same name. Papers that are not in English are moved to `data/raw/rejected/`. A summary of how many files were cleaned, rejected or failed is printed at the end.

### 5. Build the dataset

```bash
python scripts/build_corpus.py
```

Reads the cleaned PDFs, checks each document and writes the final dataset to `data/processed/clean_corpus.jsonl`.

### 6. Generate embeddings

```bash
python src/embeddings.py
```

Embeds the `text` field of every document with `all-MiniLM-L6-v2` and saves the vectors to `data/processed/embeddings.npy` and the matching IDs to `data/processed/embedding_ids.json`. The model is downloaded on the first run (about 90 MB) and cached after that.

To try several batch sizes first and use the fastest one:

```bash
python src/embeddings.py --benchmark
```

### 7. Load the embeddings into ChromaDB

```bash
python src/vector_store.py ingest
```

Stores every document's ID, vector, text and metadata in the ChromaDB database in `data/chroma/`. This only needs to be done once, or again after the embeddings change. To delete the collection and rebuild it from scratch:

```bash
python src/vector_store.py ingest --reset
```

### 8. Search

```bash
python src/vector_store.py search "making large language models safer"
python src/vector_store.py search "graph neural networks for molecules" -k 10 --min-score 0.3
```

Prints the top results with their similarity score and the time the search took. `-k` sets the number of results and `--min-score` hides results below a score.

### 9. Test retrieval performance

```bash
python scripts/test_retrieval.py
```

Runs a fixed set of test queries and measures latency and retrieval quality. Results are written to `logs/retrieval.log` and `logs/retrieval_report.json`.

### 10. Check the LLM connection

```bash
python src/llm.py --list-models
python src/llm.py --ping
```

`--list-models` prints the model IDs that are live for your API key and warns if `LLM_MODEL` is not one of them. Free model IDs change often, so run this first whenever a request fails with "model not found". `--ping` sends a short test request and prints the reply, latency and token usage.

### 11. Ask questions

```bash
python src/rag.py "What methods are used to make large language models safer?"
```

Or start interactive mode, which loads the models once and takes one question after another until an empty line is entered:

```bash
python src/rag.py
```

Each answer is followed by the list of source papers with their scores, and by the model name, latency and token usage of the LLM call.

## Text cleaning

The functions are in `src/preprocessing.py` and are applied in this order:

1. `strip_html` removes HTML tags and decodes entities such as `&`. It only parses the text as HTML when it finds a real tag like `<p>` or `<br>`, so math in abstracts such as `x<y` is left alone.
2. `normalize_unicode` applies NFKC normalization, which turns ligatures like `ﬁ` into `fi` and non-breaking spaces into normal spaces. It also removes zero-width and control characters.
3. `clean_whitespace` collapses line breaks, tabs and repeated spaces into single spaces.
4. `detect_language` identifies the language with langdetect. Only English documents are kept.

`clean_text` runs the first three steps together.

## Dataset

The cleaned dataset is at:

```
data/processed/clean_corpus.jsonl
```

It has one JSON object per line, one line per paper:

| Field           | Description                                          |
| --------------- | ---------------------------------------------------- |
| `id`          | 12-character hash of the text, the same on every run |
| `title`       | paper title                                          |
| `abstract`    | paper abstract                                       |
| `text`        | title and abstract together                          |
| `lang`        | detected language, always `en`                     |
| `n_words`     | number of words in `text`                          |
| `source_file` | name of the PDF in `data/raw/`                     |

Example:

```json
{"id": "3f9a1c2b7d4e", "title": "Example Paper Title", "abstract": "We propose a method that...", "text": "Example Paper Title We propose a method that...", "lang": "en", "n_words": 187, "source_file": "Example Paper Title.pdf"}
```

A document is left out of the dataset if it is empty, has fewer than 20 words, is not in English, would still change if cleaned again, or has the same text as a document already added. After writing the file, the script reads it back and checks that every line has the expected fields, that no ID appears twice and that the number of lines is correct.

To load it:

```python
import pandas as pd

df = pd.read_json("data/processed/clean_corpus.jsonl", lines=True)
```

## Embeddings

`src/embeddings.py` uses `all-MiniLM-L6-v2` from sentence-transformers. It produces 384-dimensional vectors, runs well on CPU and reads up to 256 tokens per text. Vectors are normalized to unit length so that similarity can be compared with a simple dot product.

Before timing starts, the model encodes a couple of texts as a warm-up, so one-time setup cost does not affect the numbers. Each run logs the following to the console and to `logs/embeddings.log`:

- model load time and device
- how many texts are longer than 256 tokens and will be truncated
- total time, texts per second and milliseconds per text

| File                                  | Contents                                                 |
| ------------------------------------- | -------------------------------------------------------- |
| `data/processed/embeddings.npy`     | array of shape `(number of documents, 384)`            |
| `data/processed/embedding_ids.json` | document IDs, in the same order as the rows of the array |

## Vector store

`src/vector_store.py` stores the embeddings in a persistent ChromaDB collection named `arxiv_papers` in `data/chroma/`. ChromaDB does not compute any embeddings itself. Both documents and queries are embedded with the same model in our own code, and the collection records which model it was built with, so a search with a different model is refused.

Each record contains the document ID, its vector, its text and this metadata: `doc_id`, `title`, `n_words` and `source_file`.

The collection uses ChromaDB's default squared L2 distance. Because the vectors are normalized, this gives the same ranking as cosine similarity, and the score shown in search results is the cosine similarity, calculated as `1 - distance / 2`.

Records are written with `upsert` in batches of 1,000, so running ingest twice does not create duplicates. Input is checked before anything is written: missing files, mismatched counts, duplicate or empty IDs, NaN values in vectors, wrong vector sizes and metadata types ChromaDB does not accept are all caught with a clear message. The full list of edge cases and how each one is handled is in `docs/chunking_and_vector_store.md`.

## Retrieval testing

`scripts/test_retrieval.py` loads the model once, runs three warm-up searches and then runs three tests:

1. **Latency by query.** Short, medium, long, non-English and meaningless queries, each run 5 times. Time is split into query encoding and the ChromaDB lookup.
2. **Latency by k.** The same query with 1, 5, 10, 50 and 100 results.
3. **Quality check.** The titles of 200 random papers are used as queries, and the script checks whether each paper comes back. It reports Recall@1, Recall@5 and MRR.

Options:

```bash
python scripts/test_retrieval.py -k 10 --repeats 10 --samples 500
```

Results from the first run on CPU with 5,007 documents:

| Metric                                 | Value      |
| -------------------------------------- | ---------- |
| Median latency                         | 14.61 ms   |
| 95th percentile latency                | 18.21 ms   |
| Share of time spent encoding the query | about 80%  |
| ChromaDB lookup                        | about 2 ms |
| Recall@1                               | 0.985      |
| Recall@5                               | 1.000      |
| MRR                                    | 0.990      |

Relevant queries scored between 0.44 and 0.64, while meaningless queries such as `????` scored between 0.17 and 0.21. A `--min-score` of 0.3 filters out most junk results.

The quality check is a sanity check. The title is part of each document's text, so title searches are expected to find the right paper almost every time. A low score would mean the index is out of date or was built with another model.

## Answer generation

Questions are answered in three steps:

1. **Retrieve.** The question is embedded and the five most similar papers are fetched from ChromaDB. Papers scoring below 0.3 are dropped.
2. **Build the prompt.** The papers are numbered and placed in the prompt as context, together with the question.
3. **Generate.** The LLM writes an answer based only on that context and cites the papers it used as `[1]`, `[2]` and so on.

If no paper scores above 0.3, the LLM is not called and the answer says that no relevant papers were found. This avoids an API call that could only produce an answer without any supporting papers.

### Files

| File               | Purpose                                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/llm.py`     | Client for the LLM API. Reads its settings from `.env`, sends chat requests and returns the answer with latency and token usage. |
| `src/prompts.py` | System prompt and formatting of the retrieved papers into the prompt                                                               |
| `src/rag.py`     | Runs the three steps above. Loads the embedding model, ChromaDB and the LLM client once.                                           |

### Provider and model

The client uses the `openai` Python package, which works with any provider that offers an OpenAI-compatible API. Groq and OpenRouter both do, so switching providers only means changing `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `.env`.

The default model is `openai/gpt-oss-20b` on Groq. It is listed as a production model and is available on the free tier, while several models used in older tutorials are now enterprise-only.

`gpt-oss-20b` is a reasoning model, which spends tokens on reasoning before it answers. `LLM_REASONING_EFFORT=low` keeps that short, which makes answers faster and uses less of the free tier's limit of 8,000 tokens per minute. For a model that does not support reasoning, leave the setting empty.

To use OpenRouter instead:

```
LLM_API_KEY=sk-or-your_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=a-model-id:free
LLM_REASONING_EFFORT=
```

### Settings

| Setting                  | Default                            | Description                                            |
| ------------------------ | ---------------------------------- | ------------------------------------------------------ |
| `LLM_API_KEY`          | none                               | API key for the provider                               |
| `LLM_BASE_URL`         | `https://api.groq.com/openai/v1` | Provider's API address                                 |
| `LLM_MODEL`            | `openai/gpt-oss-20b`             | Model ID                                               |
| `LLM_REASONING_EFFORT` | `low`                            | `low`, `medium`, `high`, or empty to turn it off |
| `LLM_MAX_TOKENS`       | `1024`                           | Maximum length of the answer, including reasoning      |
| `LLM_TEMPERATURE`      | `0.2`                            | Kept low so answers stay close to the context          |

A request for a model ID that does not exist gives a message pointing to `--list-models`, and a rejected API key gives a message pointing to `.env`.

### First results

In the first test, greetings and unrelated inputs such as `hi`, `who are u` and `csc` scored below 0.3, so no LLM call was made. The query `sd` matched one paper at 0.341, and the model asked for clarification instead of guessing. That LLM call took 754 ms and used 437 tokens in total.

```

## Running the pipeline

Run these from the project root with the virtual environment active, in this order.

### 1. Check the environment

```bash
python scripts/verify_env.py
```

Prints the PyTorch version and whether CUDA is available. A CPU-only machine will print "CUDA is not available", which is fine for this stage.

### 2. Download the papers

```bash
python scripts/fetch_arxiv.py
```

Uses the arXiv API to get the 5,000 most recent papers from cs.AI, cs.CL and cs.LG. For each paper it creates a one-page PDF with the title and abstract and saves it to `data/raw/`, named after the paper title. Characters that Windows does not allow in file names are removed, and long titles are cut to 150 characters. If two papers have the same title, the arXiv ID is added to the second file name.

This takes a few minutes. Files that already exist are skipped, so the script can be run again if it stops.

### 3. Run the tests

```bash
python -m pytest tests -v
```

Tests the cleaning functions before they are used on the data. All tests should pass before moving on.

### 4. Clean the PDFs

Back up the raw data first, since this step overwrites the files:

```powershell
Copy-Item data\raw data\raw_backup -Recurse
```

Then run:

```bash
python src/preprocessing.py
```

Each PDF in `data/raw/` is read, cleaned and saved again under the same name. Papers that are not in English are moved to `data/raw/rejected/`. A summary of how many files were cleaned, rejected or failed is printed at the end.

### 5. Build the dataset

```bash
python scripts/build_corpus.py
```

Reads the cleaned PDFs, checks each document and writes the final dataset to `data/processed/clean_corpus.jsonl`.

## Text cleaning

The functions are in `src/preprocessing.py` and are applied in this order:

1. `strip_html` removes HTML tags and decodes entities such as `&amp;`. It only parses the text as HTML when it finds a real tag like `<p>` or `<br>`, so math in abstracts such as `x<y` is left alone.
2. `normalize_unicode` applies NFKC normalization, which turns ligatures like `ﬁ` into `fi` and non-breaking spaces into normal spaces. It also removes zero-width and control characters.
3. `clean_whitespace` collapses line breaks, tabs and repeated spaces into single spaces.
4. `detect_language` identifies the language with langdetect. Only English documents are kept.

`clean_text` runs the first three steps together.

## Dataset

The final dataset is at:

```
data/processed/clean_corpus.jsonl
```

It has one JSON object per line, one line per paper:

| Field           | Description                                          |
| --------------- | ---------------------------------------------------- |
| `id`          | 12-character hash of the text, the same on every run |
| `title`       | paper title                                          |
| `abstract`    | paper abstract                                       |
| `text`        | title and abstract together                          |
| `lang`        | detected language, always`en`                      |
| `n_words`     | number of words in`text`                           |
| `source_file` | name of the PDF in`data/raw/`                      |

Example:

```json
{"id": "3f9a1c2b7d4e", "title": "Example Paper Title", "abstract": "We propose a method that...", "text": "Example Paper Title We propose a method that...", "lang": "en", "n_words": 187, "source_file": "Example Paper Title.pdf"}
```

A document is left out of the dataset if it is empty, has fewer than 20 words, is not in English, would still change if cleaned again, or has the same text as a document already added. After writing the file, the script reads it back and checks that every line has the expected fields, that no ID appears twice and that the number of lines is correct.

To load it:

```python
import pandas as pd

df = pd.read_json("data/processed/clean_corpus.jsonl", lines=True)
```
