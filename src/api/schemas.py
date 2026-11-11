"""Request and response models for the API. Invalid input gets a 422 automatically."""
from pydantic import BaseModel, Field


# ---------- requests ----------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="What to search for")
    k: int = Field(5, ge=1, le=50, description="Number of results")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Hide results below this score")
    topic: int | None = Field(None, ge=-1, description="Only this topic (see GET /topics)")
    entity: str | None = Field(None, max_length=100, description="Only papers mentioning this, e.g. Qwen3")
    exclude_reassigned: bool = Field(False, description="Skip papers whose topic came from outlier reassignment")

    model_config = {"json_schema_extra": {"examples": [
        {"query": "reward models for alignment", "k": 5, "topic": 0},
    ]}}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Your question")
    k: int = Field(5, ge=1, le=10, description="Number of papers given to the LLM")
    topic: int | None = Field(None, ge=-1)
    entity: str | None = Field(None, max_length=100)

    model_config = {"json_schema_extra": {"examples": [
        {"question": "How are LLM agents evaluated?"},
    ]}}


# ---------- responses ----------

class SearchResult(BaseModel):
    doc_id: str
    title: str
    score: float
    topic: int | None = None
    topic_label: str = ""
    entities: str = ""
    snippet: str


class SearchResponse(BaseModel):
    query: str
    count: int
    latency_ms: float
    results: list[SearchResult]


class Source(BaseModel):
    n: int
    doc_id: str
    title: str
    score: float


class Latency(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class Tokens(BaseModel):
    prompt: int | None
    completion: int | None


class AskResponse(BaseModel):
    question: str
    answer: str
    llm_called: bool
    model: str | None
    sources: list[Source]
    latency: Latency
    tokens: Tokens | None


class Topic(BaseModel):
    topic: int
    papers: int
    name: str


class HealthResponse(BaseModel):
    status: str
    documents: int
    embedding_model: str
    llm_model: str
    llm_configured: bool