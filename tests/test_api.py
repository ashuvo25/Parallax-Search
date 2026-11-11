"""Unit tests for the FastAPI endpoints.

The real pipeline (embedding model, ChromaDB, LLM) is replaced with fakes, so the
tests are fast, free and deterministic, and can force errors like 429 or 500.
"""
import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import main as api_main  # noqa: E402
from src.api.main import app, get_pipeline  # noqa: E402
from src.api.schemas import Topic  # noqa: E402
from src.llm import LLMRateLimited, LLMResponse, LLMTimeout  # noqa: E402
from src.rag import LLMNotConfigured  # noqa: E402

TOPICS = [
    Topic(topic=-1, papers=201, name="Unassigned"),
    Topic(topic=0, papers=759, name="LLM Agents"),
    Topic(topic=1, papers=383, name="Vision and Video"),
]

SEARCH_RESULTS = [
    {"id": "d1", "doc_id": "d1", "title": "Paper One", "score": 0.61, "topic": 0,
     "topic_label": "LLM Agents", "entities": "OpenAI (ORG)",
     "text": "Paper One This paper studies how agents are evaluated."},
    {"id": "d2", "doc_id": "d2", "title": "Paper Two", "score": 0.55, "topic": 0,
     "topic_label": "LLM Agents", "entities": "",
     "text": "Paper Two A benchmark for tool-using agents."},
]


# ---------- fakes ----------

class FakeCollection:
    name = "fake"

    def __init__(self, has_metadata: bool = True):
        self.has_metadata = has_metadata

    def count(self) -> int:
        return 3

    def get(self, limit=None, include=None):
        meta = {"topic": 0, "title": "x"} if self.has_metadata else {"title": "x"}
        return {"metadatas": [meta]}


class FakePipeline:
    """Stands in for RAGPipeline. Set answer_error or no_results to change behavior."""

    def __init__(self):
        self.collection = FakeCollection()
        self.embedder = object()
        self.llm = object()
        self.answer_error: Exception | None = None
        self.no_results = False
        self.last_answer_call = None
        self.last_search_call = None

    def answer(self, question, k=None, where=None):
        self.last_answer_call = {"question": question, "k": k, "where": where}
        if self.answer_error:
            raise self.answer_error
        if self.no_results:
            return {"answer": "I couldn't find any relevant papers for this question.",
                    "sources": [], "llm": None,
                    "retrieval_ms": 4.0, "generation_ms": 0.0, "total_ms": 4.5}
        llm = LLMResponse(text="Agents are evaluated with benchmarks [1].", model="fake-model",
                          latency_ms=100.0, prompt_tokens=50, completion_tokens=20,
                          finish_reason="stop")
        return {
            "answer": llm.text,
            "sources": [{"n": 1, "title": "Paper One", "score": 0.61, "doc_id": "d1"}],
            "llm": llm,
            "retrieval_ms": 5.0,
            "generation_ms": 100.0,
            "total_ms": 106.0,
        }


class ListHandler(logging.Handler):
    """Collects log records so a test can inspect them."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


# ---------- fixtures ----------

@pytest.fixture(autouse=True)
def quiet_logs(monkeypatch):
    """Keep test requests out of the real logs/api.jsonl."""
    monkeypatch.setattr(logging.getLogger("api"), "handlers", [logging.NullHandler()])


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def client(pipeline, monkeypatch):
    def fake_search(collection, embedder, query, k, min_score, where):
        pipeline.last_search_call = {"query": query, "k": k, "min_score": min_score, "where": where}
        return SEARCH_RESULTS[:k]

    monkeypatch.setattr(api_main, "search", fake_search)
    monkeypatch.setattr(api_main, "read_topics", lambda: TOPICS)
    app.dependency_overrides[get_pipeline] = lambda: pipeline

    # no `with` block, so the startup step that loads the real models never runs
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def error_of(resp) -> dict:
    body = resp.json()
    assert "error" in body, f"not in the error format: {body}"
    return body["error"]


# ---------- /health and /topics ----------

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents"] == 3
    assert body["llm_configured"] is True


def test_health_reports_missing_llm(client, pipeline):
    pipeline.llm = None
    assert client.get("/health").json()["llm_configured"] is False


def test_topics(client):
    resp = client.get("/topics")
    assert resp.status_code == 200
    assert [t["topic"] for t in resp.json()] == [-1, 0, 1]


# ---------- /search ----------

def test_search_returns_results(client):
    resp = client.post("/search", json={"query": "agent evaluation", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    first = body["results"][0]
    assert first["doc_id"] == "d1"
    assert first["snippet"].startswith("This paper studies")  # title removed from snippet


def test_search_passes_filters(client, pipeline):
    resp = client.post("/search", json={"query": "tools", "topic": 0, "entity": "OpenAI"})
    assert resp.status_code == 200
    where = pipeline.last_search_call["where"]
    assert {"topic": 0} in where["$and"]
    assert {"ent_openai": True} in where["$and"]


def test_search_strips_query(client, pipeline):
    client.post("/search", json={"query": "  agents  "})
    assert pipeline.last_search_call["query"] == "agents"


def test_search_blank_query_is_400(client):
    resp = client.post("/search", json={"query": "   "})
    assert resp.status_code == 400
    assert error_of(resp)["type"] == "bad_request"


def test_search_unknown_topic_is_400(client):
    resp = client.post("/search", json={"query": "agents", "topic": 999})
    assert resp.status_code == 400
    assert "999" in error_of(resp)["message"]


def test_search_blank_entity_is_400(client):
    resp = client.post("/search", json={"query": "agents", "entity": "   "})
    assert resp.status_code == 400


def test_search_filter_without_metadata_is_503(client, pipeline):
    pipeline.collection = FakeCollection(has_metadata=False)
    resp = client.post("/search", json={"query": "agents", "topic": 0})
    assert resp.status_code == 503


def test_search_missing_query_is_422(client):
    resp = client.post("/search", json={})
    assert resp.status_code == 422
    error = error_of(resp)
    assert error["type"] == "validation_error"
    assert error["details"][0]["field"] == "query"


@pytest.mark.parametrize("bad_k", [0, 51, "five"])
def test_search_bad_k_is_422(client, bad_k):
    resp = client.post("/search", json={"query": "agents", "k": bad_k})
    assert resp.status_code == 422
    assert error_of(resp)["details"][0]["field"] == "k"


# ---------- /ask ----------

def test_ask_returns_answer(client, pipeline):
    resp = client.post("/ask", json={"question": "How are agents evaluated?", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Agents are evaluated with benchmarks [1]."
    assert body["llm_called"] is True
    assert body["model"] == "fake-model"
    assert body["sources"][0]["doc_id"] == "d1"
    assert body["latency"] == {"retrieval_ms": 5.0, "generation_ms": 100.0, "total_ms": 106.0}
    assert body["tokens"] == {"prompt": 50, "completion": 20}
    assert pipeline.last_answer_call["k"] == 3


def test_ask_without_results_skips_llm(client, pipeline):
    pipeline.no_results = True
    body = client.post("/ask", json={"question": "hi"}).json()
    assert body["llm_called"] is False
    assert body["sources"] == []
    assert body["tokens"] is None


def test_ask_blank_question_is_400(client):
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_ask_broken_json_is_400(client):
    resp = client.post("/ask", content=b'{"question": "hi"',
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert "JSON" in error_of(resp)["message"]


def test_ask_too_long_is_422(client):
    resp = client.post("/ask", json={"question": "x" * 1001})
    assert resp.status_code == 422


def test_ask_llm_not_configured_is_503(client, pipeline):
    pipeline.answer_error = LLMNotConfigured("LLM_API_KEY is not set.")
    resp = client.post("/ask", json={"question": "agents?"})
    assert resp.status_code == 503
    assert error_of(resp)["type"] == "service_unavailable"


def test_ask_rate_limited_is_429_with_retry_after(client, pipeline):
    pipeline.answer_error = LLMRateLimited("Rate limit reached.", retry_after="12")
    resp = client.post("/ask", json={"question": "agents?"})
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "12"


def test_ask_timeout_is_504(client, pipeline):
    pipeline.answer_error = LLMTimeout("The LLM did not answer in time.")
    resp = client.post("/ask", json={"question": "agents?"})
    assert resp.status_code == 504
    assert error_of(resp)["type"] == "gateway_timeout"


def test_ask_unexpected_error_is_500_without_details(client, pipeline):
    pipeline.answer_error = RuntimeError("database password is secret123")
    resp = client.post("/ask", json={"question": "agents?"})
    assert resp.status_code == 500
    assert error_of(resp)["type"] == "internal_error"
    assert "secret123" not in resp.text  # internals never reach the user


# ---------- general behavior ----------

def test_unknown_path_is_404_in_error_format(client):
    resp = client.get("/nothing")
    assert resp.status_code == 404
    assert error_of(resp)["type"] == "not_found"


def test_wrong_method_is_405(client):
    resp = client.get("/ask")
    assert resp.status_code == 405
    assert error_of(resp)["type"] == "method_not_allowed"


def test_request_id_is_generated(client):
    resp = client.get("/health")
    assert len(resp.headers["X-Request-ID"]) == 12


def test_request_id_is_echoed(client):
    resp = client.get("/health", headers={"X-Request-ID": "my-id-123"})
    assert resp.headers["X-Request-ID"] == "my-id-123"


def test_error_contains_request_id(client):
    resp = client.post("/ask", json={"question": "   "})
    assert error_of(resp)["request_id"] == resp.headers["X-Request-ID"]


def test_request_is_logged_without_question_text(client):
    handler = ListHandler()
    logging.getLogger("api").addHandler(handler)

    client.post("/ask", json={"question": "my private question"})

    records = [r for r in handler.records if r.name == "api.requests"]
    assert len(records) == 1
    fields = records[0].fields
    assert fields["path"] == "/ask"
    assert fields["status"] == 200
    assert "latency_ms" in fields and "request_id" in fields
    assert fields["question_chars"] == len("my private question")
    assert "my private question" not in str(fields)  # the text itself is never logged