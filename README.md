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

| Field | Description |
|---|---|
| `id` | 12-character hash of the text, the same on every run |
| `title` | paper title |
| `abstract` | paper abstract |
| `text` | title and abstract together |
| `lang` | detected language, always `en` |
| `n_words` | number of words in `text` |
| `source_file` | name of the PDF in `data/raw/` |

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

| File | Contents |
|---|---|
| `data/processed/embeddings.npy` | array of shape `(number of documents, 384)` |
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

| Metric | Value |
|---|---|
| Median latency | 14.61 ms |
| 95th percentile latency | 18.21 ms |
| Share of time spent encoding the query | about 80% |
| ChromaDB lookup | about 2 ms |
| Recall@1 | 0.985 |
| Recall@5 | 1.000 |
| MRR | 0.990 |

Relevant queries scored between 0.44 and 0.64, while meaningless queries such as `????` scored between 0.17 and 0.21. A `--min-score` of 0.3 filters out most junk results.

The quality check is a sanity check. The title is part of each document's text, so title searches are expected to find the right paper almost every time. A low score would mean the index is out of date or was built with another model.

## Answer generation

Questions are answered in three steps:

1. **Retrieve.** The question is embedded and the five most similar papers are fetched from ChromaDB. Papers scoring below 0.3 are dropped.
2. **Build the prompt.** The papers are numbered and placed in the prompt as context, together with the question.
3. **Generate.** The LLM writes an answer based only on that context and cites the papers it used as `[1]`, `[2]` and so on.

If no paper scores above 0.3, the LLM is not called and the answer says that no relevant papers were found. This avoids an API call that could only produce an answer without any supporting papers.

### Files

| File | Purpose |
|---|---|
| `src/llm.py` | Client for the LLM API. Reads its settings from `.env`, sends chat requests and returns the answer with latency and token usage. |
| `src/prompts.py` | System prompt and formatting of the retrieved papers into the prompt |
| `src/rag.py` | Runs the three steps above. Loads the embedding model, ChromaDB and the LLM client once. |

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

| Setting | Default | Description |
|---|---|---|
| `LLM_API_KEY` | none | API key for the provider |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Provider's API address |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model ID |
| `LLM_REASONING_EFFORT` | `low` | `low`, `medium`, `high`, or empty to turn it off |
| `LLM_MAX_TOKENS` | `1024` | Maximum length of the answer, including reasoning |
| `LLM_TEMPERATURE` | `0.2` | Kept low so answers stay close to the context |

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
