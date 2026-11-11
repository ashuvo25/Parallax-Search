"""Parallax-Search API: semantic search and question answering over arXiv papers.

Run from the project root:
    uvicorn src.api.main:app --reload --no-access-log
"""
import csv
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.errors import register_error_handlers
from src.api.logging_config import setup_logging
from src.api.middleware import log_requests
from src.api.schemas import (AskRequest, AskResponse, HealthResponse, Latency,
                             SearchRequest, SearchResponse, SearchResult, Source,
                             Tokens, Topic)
from src.embeddings import MODEL_NAME as EMBEDDING_MODEL
from src.llm import MODEL as LLM_MODEL
from src.llm import get_client as get_llm_client
from src.rag import RAGPipeline
from src.vector_store import (TOPIC_INFO_FILE, has_nlp_metadata, load_topic_names,
                              make_where, search)

log = setup_logging()


# ---------- startup ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models once when the server starts, not on every request."""
    log.info("Loading pipeline")
    pipeline = RAGPipeline(load_llm=False)
    try:
        pipeline.llm = get_llm_client()
    except RuntimeError as e:
        log.warning(f"LLM not configured ({e}). /search works, /ask will return 503.")
    pipeline.embedder.encode(["warm up"], normalize_embeddings=True)  # first call is slow
    app.state.pipeline = pipeline
    log.info("Ready", extra={"fields": {"documents": pipeline.collection.count(),
                                        "llm_configured": pipeline.llm is not None}})
    yield


app = FastAPI(
    title="Parallax-Search API",
    version="1.0.0",
    description="Semantic search and question answering over 5,000 recent arXiv AI papers.",
    lifespan=lifespan,
)
register_error_handlers(app)
app.middleware("http")(log_requests)


def get_pipeline(request: Request) -> RAGPipeline:
    """Tests replace this with a fake pipeline."""
    return request.app.state.pipeline


# ---------- helpers ----------

def read_topics() -> list[Topic]:
    if not TOPIC_INFO_FILE.exists():
        return []
    names = load_topic_names()
    with open(TOPIC_INFO_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        Topic(topic=int(r["Topic"]), papers=int(r["Count"]),
              name=names.get(int(r["Topic"]), r["CustomName"]))
        for r in rows
    ]


def build_where(pipeline: RAGPipeline, topic: int | None, entity: str | None,
                exclude_reassigned: bool = False) -> dict | None:
    """Turn request filters into a ChromaDB filter, rejecting ones that can't work."""
    if topic is None and entity is None and not exclude_reassigned:
        return None

    if not has_nlp_metadata(pipeline.collection):
        raise HTTPException(503, "Topic and entity metadata is not loaded. "
                                 "Run: python src/vector_store.py ingest --reset")

    if topic is not None:
        known = {t.topic for t in read_topics()}
        if known and topic not in known:
            raise HTTPException(400, f"Topic {topic} does not exist. See GET /topics.")

    if entity is not None and not entity.strip():
        raise HTTPException(400, "Entity must not be empty.")

    return make_where(topic, entity.strip() if entity else None, exclude_reassigned)


def snippet(title: str, text: str, length: int = 300) -> str:
    body = text[len(title):].strip() if title and text.startswith(title) else text
    return body[:length]


# ---------- endpoints ----------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(pipeline: RAGPipeline = Depends(get_pipeline)):
    return HealthResponse(
        status="ok",
        documents=pipeline.collection.count(),
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        llm_configured=pipeline.llm is not None,
    )


@app.get("/topics", response_model=list[Topic], tags=["search"])
def topics():
    return read_topics()


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search_papers(req: SearchRequest, request: Request,
                  pipeline: RAGPipeline = Depends(get_pipeline)):
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be blank.")

    where = build_where(pipeline, req.topic, req.entity, req.exclude_reassigned)

    start = time.perf_counter()
    results = search(pipeline.collection, pipeline.embedder, query, req.k, req.min_score, where)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    request.state.log_extra.update(
        query_chars=len(query), k=req.k, filtered=where is not None,
        results=len(results), search_ms=latency_ms,
    )
    return SearchResponse(
        query=query,
        count=len(results),
        latency_ms=latency_ms,
        results=[
            SearchResult(
                doc_id=r["doc_id"],
                title=r["title"],
                score=r["score"],
                topic=r.get("topic"),
                topic_label=r.get("topic_label", ""),
                entities=r.get("entities", ""),
                snippet=snippet(r["title"], r["text"]),
            )
            for r in results
        ],
    )


@app.post("/ask", response_model=AskResponse, tags=["answer"])
def ask(req: AskRequest, request: Request, pipeline: RAGPipeline = Depends(get_pipeline)):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question must not be blank.")

    where = build_where(pipeline, req.topic, req.entity)
    result = pipeline.answer(question, k=req.k, where=where)  # LLM errors -> llm_error_handler

    llm = result["llm"]
    request.state.log_extra.update(
        question_chars=len(question), k=req.k, filtered=where is not None,
        llm_called=llm is not None, sources=len(result["sources"]),
        retrieval_ms=result["retrieval_ms"], generation_ms=result["generation_ms"],
        prompt_tokens=llm.prompt_tokens if llm else None,
        completion_tokens=llm.completion_tokens if llm else None,
    )
    return AskResponse(
        question=question,
        answer=result["answer"],
        llm_called=llm is not None,
        model=llm.model if llm else None,
        sources=[Source(**s) for s in result["sources"]],
        latency=Latency(
            retrieval_ms=result["retrieval_ms"],
            generation_ms=result["generation_ms"],
            total_ms=result["total_ms"],
        ),
        tokens=Tokens(prompt=llm.prompt_tokens, completion=llm.completion_tokens) if llm else None,
    )