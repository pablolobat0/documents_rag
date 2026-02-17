from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class EvalSample:
    question: str
    doc_id: str
    section_id: int
    context: str
    title: str


@dataclass
class RetrievalResult:
    question: str
    doc_id: str
    relevant_positions: list[int] = field(default_factory=list)
    total_relevant: int = 0
    latency_ms: float = 0.0


@dataclass
class RetrievalMetrics:
    recall_at_25: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_queries: int = 0


class JudgeScore(BaseModel):
    reasoning: str
    score: int = Field(ge=0, le=5)
