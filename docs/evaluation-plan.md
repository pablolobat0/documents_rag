# LLM-as-a-Judge Evaluation System — Implementation Plan

## Context

The RAG system currently has no way to measure answer quality. This plan adds an automated evaluation pipeline that:

1. Generates synthetic Q&A pairs from ingested documents
2. Runs each question through the full RAG pipeline (retrieve + generate)
3. Uses a separate LLM judge to score three dimensions: **faithfulness**, **answer relevance**, and **context relevance**
4. Outputs a JSON report with per-sample scores and aggregate statistics

**Entry point**: CLI script (`evaluate.py`)
**Judge model**: Separately configurable via `JUDGE_MODEL` env var (can be stronger than the RAG model)

---

## Architecture Overview

The evaluation system follows the same DDD layers as the existing codebase:

```
CLI entry point (evaluate.py)
    │
    ▼
Application Layer (use cases + DTOs)
    │
    ▼
Domain Layer (value objects, ports, prompts)
    ▲
    │
Infrastructure Layer (implementations)
```

### Pipeline Flow

```
Phase 1: Synthetic Dataset Generation
┌─────────────────────────────────────────────────────────┐
│ Qdrant (scroll) → Random chunks → LLM generates Q&A    │
│ per chunk → SyntheticDataset(samples)                   │
└─────────────────────────────────────────────────────────┘

Phase 2: Evaluation
┌─────────────────────────────────────────────────────────┐
│ For each sample:                                        │
│   1. vector_store.search(question) → retrieved contexts │
│   2. agent.run(question) → generated answer             │
│   3. Judge LLM scores:                                  │
│      - Faithfulness (answer vs contexts)                │
│      - Answer Relevance (answer vs question)            │
│      - Context Relevance (contexts vs question)         │
│ → EvaluationReport (per-sample + aggregate)             │
└─────────────────────────────────────────────────────────┘
```

### LLM Calls Per Sample

| Step | LLM Calls | Description |
|------|-----------|-------------|
| Q&A Generation | 1 | Generate question from chunk |
| Agent Run | ~3 | Search decision + re-rank + response generation |
| Judge: Faithfulness | 1 | Score answer groundedness |
| Judge: Answer Relevance | 1 | Score answer relevance |
| Judge: Context Relevance | 1 | Score retrieval quality |
| **Total per sample** | **~7** | |

With 20 samples (default): ~140 LLM calls total.

---

## Files to Create

### 1. Domain Layer

#### `src/domain/value_objects/evaluation.py`

All frozen dataclasses, framework-agnostic.

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SyntheticSample:
    """A single synthetic Q&A sample generated from a document chunk."""
    question: str
    ground_truth_context: str
    source_document: str | None = None  # document_name from chunk metadata


@dataclass(frozen=True)
class SyntheticDataset:
    """Collection of synthetic Q&A samples."""
    samples: list[SyntheticSample]

    @property
    def size(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single evaluation dimension."""
    score: float          # 0.0 to 1.0
    reasoning: str        # Judge's explanation


@dataclass(frozen=True)
class EvaluationResult:
    """Result of evaluating a single sample through the full pipeline."""
    question: str
    ground_truth_context: str
    retrieved_contexts: list[str]
    generated_answer: str
    faithfulness: DimensionScore
    answer_relevance: DimensionScore
    context_relevance: DimensionScore


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate statistics across all evaluation results."""
    total_samples: int
    faithfulness_mean: float
    faithfulness_min: float
    faithfulness_max: float
    answer_relevance_mean: float
    answer_relevance_min: float
    answer_relevance_max: float
    context_relevance_mean: float
    context_relevance_min: float
    context_relevance_max: float


@dataclass(frozen=True)
class EvaluationReport:
    """Complete evaluation output: per-sample results + aggregate summary."""
    results: list[EvaluationResult]
    summary: EvaluationSummary
```

---

#### `src/domain/ports/document_sampler_port.py`

New Protocol for random chunk sampling. Kept separate from `VectorStorePort` because sampling is a fundamentally different concern from similarity search.

```python
from typing import Protocol

from src.domain.value_objects.retrieved_document import RetrievedDocument


class DocumentSamplerPort(Protocol):
    """Port for sampling stored documents for evaluation."""

    def sample_chunks(self, num_chunks: int) -> list[RetrievedDocument]:
        """Retrieve a random sample of document chunks from storage."""
        ...
```

---

#### `src/domain/prompts/evaluation.py`

Follows the `AgentPrompts` pattern: class with uppercase string constants and static helper methods.

```python
class EvaluationPrompts:
    """Prompts for synthetic dataset generation and LLM-as-a-Judge evaluation."""

    # ──────────────────────────────────────────────
    # Synthetic Dataset Generation
    # ──────────────────────────────────────────────

    GENERATE_QA_SYSTEM_PROMPT = """You are an expert at creating evaluation \
questions for retrieval-augmented generation (RAG) systems. Your task is to \
generate a question that a user might naturally ask, which the given document \
chunk should be able to answer.

**Requirements:**

1. The question must be answerable using ONLY the information in the provided chunk.
2. The question should sound natural, as if a real user typed it into a chat interface.
3. The question should be specific enough that a vague answer would be insufficient.
4. Do NOT ask questions about the document itself (e.g., "What does the document \
say about..."). Ask questions as a user seeking information would.
5. Do NOT include phrases like "according to the text" or "based on the passage".
6. Vary question types: factual, comparative, explanatory, procedural.

Return your response using the required structured output format."""

    GENERATE_QA_USER_PROMPT = """Generate a natural user question from this \
document chunk:

---
{chunk_content}
---"""

    # ──────────────────────────────────────────────
    # Faithfulness Evaluation
    # Does the answer contain ONLY information supported by the retrieved contexts?
    # ──────────────────────────────────────────────

    FAITHFULNESS_SYSTEM_PROMPT = """You are an impartial judge evaluating the \
faithfulness of an AI-generated answer. Faithfulness measures whether every \
claim in the answer is supported by the provided retrieved contexts.

**Scoring rubric (0.0 to 1.0):**

- **1.0**: Every claim in the answer is directly supported by the retrieved \
contexts. No hallucinated or fabricated information.
- **0.75**: Almost all claims are supported. Minor inferences are reasonable \
and grounded in context.
- **0.5**: Some claims are supported, but the answer also contains unsupported \
statements or extrapolations not justified by the contexts.
- **0.25**: Most of the answer contains information not found in the retrieved \
contexts.
- **0.0**: The answer is entirely fabricated or contradicts the retrieved contexts.

**Instructions:**

1. Read the retrieved contexts carefully.
2. Go through each claim in the answer and check if it is supported by the contexts.
3. Provide your reasoning, citing specific claims that are or are not supported.
4. Assign a score based on the rubric above.

Return your response using the required structured output format."""

    FAITHFULNESS_USER_PROMPT = """**Question:** {question}

**Retrieved Contexts:**
{contexts}

**Generated Answer:** {answer}"""

    # ──────────────────────────────────────────────
    # Answer Relevance Evaluation
    # Does the answer actually address the question asked?
    # ──────────────────────────────────────────────

    ANSWER_RELEVANCE_SYSTEM_PROMPT = """You are an impartial judge evaluating \
the relevance of an AI-generated answer to a user question. Answer relevance \
measures whether the answer directly and completely addresses what the user asked.

**Scoring rubric (0.0 to 1.0):**

- **1.0**: The answer directly and completely addresses the question. It is \
focused, on-topic, and provides the information requested.
- **0.75**: The answer addresses the question well but may include minor \
tangential information or miss a small aspect.
- **0.5**: The answer partially addresses the question but is incomplete, \
vague, or includes substantial irrelevant information.
- **0.25**: The answer barely relates to the question. Most of the response \
is off-topic or unhelpful.
- **0.0**: The answer does not address the question at all, or is a \
refusal/error message.

**Instructions:**

1. Read the question carefully to understand what is being asked.
2. Evaluate whether the answer provides what the user needs.
3. Consider completeness: does it answer all parts of the question?
4. Consider focus: does it avoid unnecessary tangents?
5. Provide your reasoning, then assign a score.

Return your response using the required structured output format."""

    ANSWER_RELEVANCE_USER_PROMPT = """**Question:** {question}

**Generated Answer:** {answer}"""

    # ──────────────────────────────────────────────
    # Context Relevance Evaluation
    # Are the retrieved contexts actually relevant to answering the question?
    # ──────────────────────────────────────────────

    CONTEXT_RELEVANCE_SYSTEM_PROMPT = """You are an impartial judge evaluating \
the relevance of retrieved contexts to a user question. Context relevance \
measures whether the retrieval system found documents that are useful for \
answering the question.

**Scoring rubric (0.0 to 1.0):**

- **1.0**: All retrieved contexts are highly relevant to the question and \
contain information needed to answer it.
- **0.75**: Most contexts are relevant. One or two may be tangentially related \
but the core information is present.
- **0.5**: About half the contexts are relevant. The retrieval included a \
significant amount of irrelevant material.
- **0.25**: Most contexts are irrelevant. Only a small portion contains \
useful information.
- **0.0**: None of the retrieved contexts are relevant to the question.

**Instructions:**

1. Read the question to understand what information is needed.
2. Examine each retrieved context for relevance to the question.
3. Consider what fraction of the contexts would actually help answer the question.
4. Provide your reasoning, then assign a score.

Return your response using the required structured output format."""

    CONTEXT_RELEVANCE_USER_PROMPT = """**Question:** {question}

**Retrieved Contexts:**
{contexts}"""

    # ──────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────

    @staticmethod
    def format_contexts(contexts: list[str]) -> str:
        """Format a list of context strings for inclusion in prompts."""
        return "\n\n".join(
            f"[Context {i + 1}]: {content}" for i, content in enumerate(contexts)
        )

    @staticmethod
    def format_faithfulness_prompt(
        question: str, contexts: list[str], answer: str
    ) -> str:
        return EvaluationPrompts.FAITHFULNESS_USER_PROMPT.format(
            question=question,
            contexts=EvaluationPrompts.format_contexts(contexts),
            answer=answer,
        )

    @staticmethod
    def format_answer_relevance_prompt(question: str, answer: str) -> str:
        return EvaluationPrompts.ANSWER_RELEVANCE_USER_PROMPT.format(
            question=question,
            answer=answer,
        )

    @staticmethod
    def format_context_relevance_prompt(
        question: str, contexts: list[str]
    ) -> str:
        return EvaluationPrompts.CONTEXT_RELEVANCE_USER_PROMPT.format(
            question=question,
            contexts=EvaluationPrompts.format_contexts(contexts),
        )

    @staticmethod
    def format_generate_qa_prompt(chunk_content: str) -> str:
        return EvaluationPrompts.GENERATE_QA_USER_PROMPT.format(
            chunk_content=chunk_content,
        )
```

**Prompt design rationale:**
- 5-point rubric (0.0, 0.25, 0.5, 0.75, 1.0) anchors scoring — more reliable than open-ended floats
- Contexts numbered `[Context 1]`, `[Context 2]` so the judge can reference them in reasoning
- Generation prompt forbids meta-references ("according to the document") to produce natural questions
- Each system prompt separates instructions from scoring rubric clearly

---

### 2. Infrastructure Layer

#### `src/infrastructure/evaluation/__init__.py`

Empty file (package marker).

---

#### `src/infrastructure/evaluation/schemas.py`

Pydantic models for structured LLM output. Follows the pattern in `src/infrastructure/agent/schemas.py`.

```python
from pydantic import BaseModel, Field


class GeneratedQA(BaseModel):
    """Structured output for synthetic Q&A generation."""
    question: str = Field(description="A natural user question answerable from the chunk")


class JudgeScore(BaseModel):
    """Structured output for a single judge evaluation dimension."""
    reasoning: str = Field(description="Step-by-step reasoning for the score")
    score: float = Field(
        description="Score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
```

`JudgeScore` is reused across all three evaluation dimensions — same schema shape, different prompts. The `Field(ge=0.0, le=1.0)` constraint ensures valid scores.

---

#### `src/infrastructure/storage/qdrant_document_sampler.py`

Implements `DocumentSamplerPort` using Qdrant's `scroll` API.

```python
import random
import logging

from qdrant_client import QdrantClient

from src.domain.value_objects.retrieved_document import RetrievedDocument

logger = logging.getLogger(__name__)


class QdrantDocumentSampler:
    """Samples random document chunks from Qdrant. Implements DocumentSamplerPort."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    def sample_chunks(self, num_chunks: int) -> list[RetrievedDocument]:
        """Retrieve a random sample of document chunks from Qdrant."""
        all_points = []
        offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            logger.warning("No documents found in collection '%s'", self._collection_name)
            return []

        sample_size = min(num_chunks, len(all_points))
        sampled = random.sample(all_points, sample_size)

        return [
            RetrievedDocument(
                page_content=point.payload.get("page_content", ""),
                metadata={
                    k: v for k, v in point.payload.items() if k != "page_content"
                },
            )
            for point in sampled
        ]
```

**Why `scroll` instead of `search`?** We need a random sample, not similarity-based retrieval. We fetch without vectors to minimize memory usage. The `QdrantClient` instance and collection name are injected — available from `container.qdrant.client` and `settings.qdrant_collection_name`.

---

#### `src/infrastructure/evaluation/dataset_generator.py`

Generates synthetic Q&A pairs from stored document chunks.

```python
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.domain.ports.document_sampler_port import DocumentSamplerPort
from src.domain.prompts.evaluation import EvaluationPrompts
from src.domain.value_objects.evaluation import SyntheticSample, SyntheticDataset
from src.infrastructure.evaluation.schemas import GeneratedQA

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """Generates synthetic Q&A datasets from stored document chunks."""

    def __init__(
        self,
        llm: BaseChatModel,
        document_sampler: DocumentSamplerPort,
    ) -> None:
        self._llm = llm
        self._document_sampler = document_sampler

    def generate(self, num_samples: int) -> SyntheticDataset:
        """Generate a synthetic dataset by sampling chunks and creating questions."""
        chunks = self._document_sampler.sample_chunks(num_samples)

        if not chunks:
            logger.warning("No chunks available for dataset generation")
            return SyntheticDataset(samples=[])

        structured_llm = self._llm.with_structured_output(GeneratedQA)
        samples: list[SyntheticSample] = []

        for i, chunk in enumerate(chunks):
            try:
                user_prompt = EvaluationPrompts.format_generate_qa_prompt(
                    chunk.page_content
                )

                result = structured_llm.invoke([
                    SystemMessage(content=EvaluationPrompts.GENERATE_QA_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ])

                if result and result.question:
                    source = (
                        chunk.metadata.get("document_name")
                        if chunk.metadata
                        else None
                    )
                    samples.append(
                        SyntheticSample(
                            question=result.question,
                            ground_truth_context=chunk.page_content,
                            source_document=source,
                        )
                    )
                    logger.info(
                        "Generated sample %d/%d: %s",
                        i + 1, len(chunks), result.question[:80],
                    )
            except Exception as e:
                logger.error("Failed to generate Q&A for chunk %d: %s", i, e)
                continue

        logger.info("Generated %d samples from %d chunks", len(samples), len(chunks))
        return SyntheticDataset(samples=samples)
```

**Design decisions:**
- Accepts `BaseChatModel` (not `LLMPort`) because it needs `with_structured_output`, matching the `LanggraphAgent.__init__` pattern
- Continues on individual failures (logs and skips) rather than aborting
- Each chunk becomes one sample; the chunk content IS the ground truth context

---

#### `src/infrastructure/evaluation/llm_judge.py`

The core judge implementation. Evaluates RAG outputs across three dimensions.

```python
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.domain.prompts.evaluation import EvaluationPrompts
from src.domain.value_objects.evaluation import DimensionScore
from src.infrastructure.evaluation.schemas import JudgeScore

logger = logging.getLogger(__name__)


class LLMJudge:
    """Evaluates RAG pipeline outputs across three dimensions using an LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._structured_llm = llm.with_structured_output(JudgeScore)

    def evaluate_faithfulness(
        self,
        question: str,
        contexts: list[str],
        answer: str,
    ) -> DimensionScore:
        """Judge whether the answer is faithful to the retrieved contexts."""
        user_prompt = EvaluationPrompts.format_faithfulness_prompt(
            question, contexts, answer
        )
        return self._evaluate(
            EvaluationPrompts.FAITHFULNESS_SYSTEM_PROMPT,
            user_prompt,
            "faithfulness",
        )

    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> DimensionScore:
        """Judge whether the answer is relevant to the question."""
        user_prompt = EvaluationPrompts.format_answer_relevance_prompt(
            question, answer
        )
        return self._evaluate(
            EvaluationPrompts.ANSWER_RELEVANCE_SYSTEM_PROMPT,
            user_prompt,
            "answer_relevance",
        )

    def evaluate_context_relevance(
        self,
        question: str,
        contexts: list[str],
    ) -> DimensionScore:
        """Judge whether the retrieved contexts are relevant to the question."""
        user_prompt = EvaluationPrompts.format_context_relevance_prompt(
            question, contexts
        )
        return self._evaluate(
            EvaluationPrompts.CONTEXT_RELEVANCE_SYSTEM_PROMPT,
            user_prompt,
            "context_relevance",
        )

    def _evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        dimension_name: str,
    ) -> DimensionScore:
        """Run a single judge evaluation and return a DimensionScore."""
        try:
            result = self._structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            return DimensionScore(
                score=result.score,
                reasoning=result.reasoning,
            )
        except Exception as e:
            logger.error("Judge failed for %s: %s", dimension_name, e)
            return DimensionScore(
                score=0.0,
                reasoning=f"Evaluation failed: {e}",
            )
```

**Design notes:**
- `_evaluate` centralizes the shared pattern across all 3 dimensions
- On failure, returns `DimensionScore(score=0.0, ...)` instead of crashing — the pipeline never stops mid-evaluation
- `_structured_llm` created once in `__init__` since it's reused (same `JudgeScore` schema, different prompts)

---

#### `src/infrastructure/evaluation/pipeline_runner.py`

Orchestrates retrieval + generation + judging for each sample.

```python
import logging
import uuid

from src.domain.ports.agent_port import AgentPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.value_objects.chat_message import ChatMessage
from src.domain.value_objects.evaluation import (
    EvaluationResult,
    EvaluationReport,
    EvaluationSummary,
    SyntheticDataset,
)
from src.infrastructure.evaluation.llm_judge import LLMJudge

logger = logging.getLogger(__name__)


class EvaluationPipelineRunner:
    """Runs the full evaluation pipeline: question → RAG → judge."""

    def __init__(
        self,
        agent: AgentPort,
        vector_store: VectorStorePort,
        judge: LLMJudge,
        num_retrieval_docs: int = 10,
    ) -> None:
        self._agent = agent
        self._vector_store = vector_store
        self._judge = judge
        self._num_retrieval_docs = num_retrieval_docs

    def run(self, dataset: SyntheticDataset) -> EvaluationReport:
        """Evaluate every sample in the dataset through the full pipeline."""
        results: list[EvaluationResult] = []

        for i, sample in enumerate(dataset.samples):
            logger.info(
                "Evaluating sample %d/%d: %s",
                i + 1, dataset.size, sample.question[:80],
            )
            try:
                result = self._evaluate_single(
                    sample.question, sample.ground_truth_context
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed to evaluate sample %d: %s", i, e)
                continue

        summary = self._compute_summary(results)
        return EvaluationReport(results=results, summary=summary)

    def _evaluate_single(
        self,
        question: str,
        ground_truth_context: str,
    ) -> EvaluationResult:
        """Run one question through the RAG pipeline and judge all dimensions."""

        # Step 1: Get retrieved contexts (separate from agent for evaluation)
        retrieved_docs = self._vector_store.search(
            question, self._num_retrieval_docs
        )
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        # Step 2: Run the full agent pipeline to get the answer
        session_id = f"eval-{uuid.uuid4().hex[:12]}"
        messages = [ChatMessage(role="user", content=question)]
        generated_answer = self._agent.run(messages, session_id)

        # Step 3: Judge all three dimensions
        faithfulness = self._judge.evaluate_faithfulness(
            question, retrieved_contexts, generated_answer
        )
        answer_relevance = self._judge.evaluate_answer_relevance(
            question, generated_answer
        )
        context_relevance = self._judge.evaluate_context_relevance(
            question, retrieved_contexts
        )

        return EvaluationResult(
            question=question,
            ground_truth_context=ground_truth_context,
            retrieved_contexts=retrieved_contexts,
            generated_answer=generated_answer,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_relevance=context_relevance,
        )

    def _compute_summary(
        self,
        results: list[EvaluationResult],
    ) -> EvaluationSummary:
        """Compute aggregate statistics across all results."""
        if not results:
            return EvaluationSummary(
                total_samples=0,
                faithfulness_mean=0.0, faithfulness_min=0.0, faithfulness_max=0.0,
                answer_relevance_mean=0.0, answer_relevance_min=0.0,
                answer_relevance_max=0.0,
                context_relevance_mean=0.0, context_relevance_min=0.0,
                context_relevance_max=0.0,
            )

        def stats(scores: list[float]) -> tuple[float, float, float]:
            return (
                round(sum(scores) / len(scores), 4),
                min(scores),
                max(scores),
            )

        f_scores = [r.faithfulness.score for r in results]
        a_scores = [r.answer_relevance.score for r in results]
        c_scores = [r.context_relevance.score for r in results]

        f_mean, f_min, f_max = stats(f_scores)
        a_mean, a_min, a_max = stats(a_scores)
        c_mean, c_min, c_max = stats(c_scores)

        return EvaluationSummary(
            total_samples=len(results),
            faithfulness_mean=f_mean,
            faithfulness_min=f_min,
            faithfulness_max=f_max,
            answer_relevance_mean=a_mean,
            answer_relevance_min=a_min,
            answer_relevance_max=a_max,
            context_relevance_mean=c_mean,
            context_relevance_min=c_min,
            context_relevance_max=c_max,
        )
```

**Key design decision — context mismatch:** The `pipeline_runner` retrieves contexts via `vector_store.search()` for the judge, but the agent does its own internal retrieval + re-ranking. This means:
- The **judge** evaluates raw retrieval quality (pre-reranking)
- The **answer** comes from re-ranked context (post-reranking, via agent)

This is acceptable for v1 and diagnostically more informative — it tells you about both retrieval and re-ranking quality. The alternative (modifying the agent to expose intermediate state) would break the `AgentPort` interface.

**Ephemeral sessions:** Each sample uses `eval-{uuid}` session ID to prevent conversation state leaking between evaluation samples.

---

### 3. Application Layer

#### `src/application/dto/evaluation_dto.py`

```python
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunEvaluationRequest:
    num_samples: int = 20
    output_path: Path = field(default_factory=lambda: Path("evaluation_results.json"))


@dataclass
class EvaluationResponse:
    success: bool
    output_path: str
    total_samples: int
    faithfulness_mean: float
    answer_relevance_mean: float
    context_relevance_mean: float
    message: str = ""
```

---

#### `src/application/use_cases/run_evaluation.py`

```python
import json
import logging
from dataclasses import asdict

from src.application.dto.evaluation_dto import (
    RunEvaluationRequest,
    EvaluationResponse,
)
from src.infrastructure.evaluation.dataset_generator import SyntheticDatasetGenerator
from src.infrastructure.evaluation.pipeline_runner import EvaluationPipelineRunner

logger = logging.getLogger(__name__)


class RunEvaluationUseCase:
    """Use case for running a complete RAG evaluation."""

    def __init__(
        self,
        dataset_generator: SyntheticDatasetGenerator,
        pipeline_runner: EvaluationPipelineRunner,
    ) -> None:
        self._dataset_generator = dataset_generator
        self._pipeline_runner = pipeline_runner

    def execute(self, request: RunEvaluationRequest) -> EvaluationResponse:
        """Generate dataset, run evaluation, write results to JSON."""
        try:
            # Phase 1: Generate synthetic dataset
            logger.info(
                "Generating synthetic dataset with %d samples...",
                request.num_samples,
            )
            dataset = self._dataset_generator.generate(request.num_samples)

            if dataset.size == 0:
                return EvaluationResponse(
                    success=False,
                    output_path=str(request.output_path),
                    total_samples=0,
                    faithfulness_mean=0.0,
                    answer_relevance_mean=0.0,
                    context_relevance_mean=0.0,
                    message="No samples could be generated. Are there documents "
                    "in the vector store?",
                )

            # Phase 2: Run evaluation pipeline
            logger.info("Running evaluation on %d samples...", dataset.size)
            report = self._pipeline_runner.run(dataset)

            # Phase 3: Write results to JSON
            report_dict = asdict(report)
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(request.output_path, "w") as f:
                json.dump(report_dict, f, indent=2, default=str)

            logger.info(
                "Evaluation complete. Results written to %s", request.output_path
            )

            return EvaluationResponse(
                success=True,
                output_path=str(request.output_path),
                total_samples=report.summary.total_samples,
                faithfulness_mean=report.summary.faithfulness_mean,
                answer_relevance_mean=report.summary.answer_relevance_mean,
                context_relevance_mean=report.summary.context_relevance_mean,
                message="Evaluation completed successfully",
            )

        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            return EvaluationResponse(
                success=False,
                output_path=str(request.output_path),
                total_samples=0,
                faithfulness_mean=0.0,
                answer_relevance_mean=0.0,
                context_relevance_mean=0.0,
                message=f"Evaluation failed: {e}",
            )
```

---

### 4. CLI Entry Point

#### `evaluate.py` (project root)

```python
"""CLI entry point for running RAG evaluation."""

import argparse
import logging
import sys
from pathlib import Path

from src.container import get_container
from src.application.dto.evaluation_dto import RunEvaluationRequest


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge evaluation on the RAG pipeline"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=20,
        help="Number of synthetic Q&A samples to generate (default: 20)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Output JSON file path (default: evaluation_results.json)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting RAG evaluation...")

    container = get_container()
    use_case = container.run_evaluation_use_case

    request = RunEvaluationRequest(
        num_samples=args.num_samples,
        output_path=args.output,
    )

    response = use_case.execute(request)

    if response.success:
        print(f"\nEvaluation Complete")
        print(f"{'=' * 50}")
        print(f"Samples evaluated: {response.total_samples}")
        print(f"Output file:       {response.output_path}")
        print(f"{'=' * 50}")
        print(f"Faithfulness:      {response.faithfulness_mean:.4f}")
        print(f"Answer Relevance:  {response.answer_relevance_mean:.4f}")
        print(f"Context Relevance: {response.context_relevance_mean:.4f}")
        print(f"{'=' * 50}")
    else:
        print(f"\nEvaluation Failed: {response.message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Usage examples:**

```bash
# Basic run (20 samples, default output)
uv run python evaluate.py

# Custom sample count and output path
uv run python evaluate.py -n 50 -o results/eval_2024.json -v

# With a stronger judge model
JUDGE_MODEL=llama3.1:70b uv run python evaluate.py -n 30
```

---

## Files to Modify

### 1. `config/settings.py`

Add to the `Settings` dataclass:

```python
# Evaluation
judge_model: str = os.getenv("JUDGE_MODEL", "")
eval_num_samples: int = int(os.getenv("EVAL_NUM_SAMPLES", "20"))
eval_retrieval_docs: int = int(os.getenv("EVAL_RETRIEVAL_DOCS", "10"))
```

`judge_model` defaults to empty string. When empty, the container falls back to `settings.model`.

---

### 2. `src/container.py`

Add new imports and 6 `cached_property` entries:

```python
# New imports
from src.infrastructure.evaluation.dataset_generator import SyntheticDatasetGenerator
from src.infrastructure.evaluation.llm_judge import LLMJudge
from src.infrastructure.evaluation.pipeline_runner import EvaluationPipelineRunner
from src.infrastructure.storage.qdrant_document_sampler import QdrantDocumentSampler
from src.application.use_cases.run_evaluation import RunEvaluationUseCase


# New properties in Container class:

@cached_property
def judge_model(self) -> ChatOllama:
    model = settings.judge_model or settings.model
    return ChatOllama(
        model=model,
        base_url=settings.ollama_url,
    )

@cached_property
def document_sampler(self) -> QdrantDocumentSampler:
    return QdrantDocumentSampler(
        client=self.qdrant.client,
        collection_name=settings.qdrant_collection_name,
    )

@cached_property
def dataset_generator(self) -> SyntheticDatasetGenerator:
    return SyntheticDatasetGenerator(
        llm=self.chat_model,
        document_sampler=self.document_sampler,
    )

@cached_property
def llm_judge(self) -> LLMJudge:
    return LLMJudge(llm=self.judge_model)

@cached_property
def evaluation_pipeline_runner(self) -> EvaluationPipelineRunner:
    return EvaluationPipelineRunner(
        agent=self.agent,
        vector_store=self.qdrant,
        judge=self.llm_judge,
        num_retrieval_docs=settings.eval_retrieval_docs,
    )

@cached_property
def run_evaluation_use_case(self) -> RunEvaluationUseCase:
    return RunEvaluationUseCase(
        dataset_generator=self.dataset_generator,
        pipeline_runner=self.evaluation_pipeline_runner,
    )
```

Note: `dataset_generator` uses `self.chat_model` (regular model) for Q&A generation, while `llm_judge` uses `self.judge_model` (separate, potentially stronger model). This separation is intentional.

---

## Tests

### Test Files to Create (6)

| File | What it tests |
|------|---------------|
| `tests/domain/test_evaluation_value_objects.py` | Construction, frozen immutability, `size` property, empty results |
| `tests/infrastructure/test_dataset_generator.py` | Happy path, empty chunks, LLM failure skip-and-continue |
| `tests/infrastructure/test_llm_judge.py` | All 3 evaluate methods, failure fallback to `score=0.0` |
| `tests/infrastructure/test_pipeline_runner.py` | Full pipeline with mocked deps, `_compute_summary` stats, empty dataset |
| `tests/infrastructure/test_qdrant_document_sampler.py` | Scroll mock, empty collection, `num_chunks > available` |
| `tests/application/test_run_evaluation.py` | Happy path (JSON written), empty dataset, exception handling |

### Mock Patterns

Following existing test conventions:
- `create_autospec` for Protocol-based ports (`DocumentSamplerPort`, `AgentPort`, `VectorStorePort`)
- `MagicMock` for LLM's `with_structured_output` return value
- Fixtures in `tests/conftest.py` for shared mocks (`mock_document_sampler`)

---

## Implementation Order

| Step | File | Dependencies |
|------|------|-------------|
| 1 | `config/settings.py` | None |
| 2 | `src/domain/value_objects/evaluation.py` | None |
| 3 | `src/domain/ports/document_sampler_port.py` | `RetrievedDocument` |
| 4 | `src/domain/prompts/evaluation.py` | None |
| 5 | `src/infrastructure/evaluation/__init__.py` | None |
| 6 | `src/infrastructure/evaluation/schemas.py` | None |
| 7 | `src/infrastructure/storage/qdrant_document_sampler.py` | Port (step 3) |
| 8 | `src/infrastructure/evaluation/dataset_generator.py` | Port, prompts, schemas, VOs |
| 9 | `src/infrastructure/evaluation/llm_judge.py` | Prompts, schemas, VOs |
| 10 | `src/infrastructure/evaluation/pipeline_runner.py` | Judge, ports, VOs |
| 11 | `src/application/dto/evaluation_dto.py` | None |
| 12 | `src/application/use_cases/run_evaluation.py` | DTOs, generator, runner |
| 13 | `src/container.py` | All infrastructure |
| 14 | `evaluate.py` | Container, DTOs |
| 15 | All test files | After source files |

---

## Example Output

The `evaluation_results.json` file:

```json
{
  "results": [
    {
      "question": "What are the main benefits of using dependency injection?",
      "ground_truth_context": "Dependency injection provides several benefits including testability, loose coupling...",
      "retrieved_contexts": [
        "Dependency injection is a design pattern...",
        "Testing with DI allows mock objects..."
      ],
      "generated_answer": "The main benefits of dependency injection include improved testability...",
      "faithfulness": {
        "score": 0.75,
        "reasoning": "The answer correctly mentions testability and loose coupling which are in the contexts. However, it also mentions 'runtime flexibility' which is not directly supported."
      },
      "answer_relevance": {
        "score": 1.0,
        "reasoning": "The answer directly addresses the question about benefits of DI, listing specific benefits with brief explanations."
      },
      "context_relevance": {
        "score": 0.75,
        "reasoning": "Both retrieved contexts are relevant to dependency injection. The first directly discusses DI patterns, the second covers testing with DI."
      }
    }
  ],
  "summary": {
    "total_samples": 20,
    "faithfulness_mean": 0.7125,
    "faithfulness_min": 0.25,
    "faithfulness_max": 1.0,
    "answer_relevance_mean": 0.8375,
    "answer_relevance_min": 0.5,
    "answer_relevance_max": 1.0,
    "context_relevance_mean": 0.65,
    "context_relevance_min": 0.25,
    "context_relevance_max": 1.0
  }
}
```

---

## Potential Challenges

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| **Context mismatch** between `vector_store.search` and agent's internal re-ranked retrieval | Judge sees pre-reranking contexts, answer uses post-reranking | Acceptable for v1; evaluates retrieval recall independently |
| **Structured output reliability** with small Ollama models | Judge may fail to produce valid JSON | `JudgeScore` schema is minimal (2 fields); `JUDGE_MODEL` allows stronger model |
| **Evaluation speed** (~7 LLM calls per sample) | 20 samples = ~140 calls, potentially slow | Moderate default count; progress logging; future: async |
| **Qdrant scroll** for large collections | All payloads loaded into memory | Fine for typical RAG (thousands of chunks); future: count + random offset |
| **Agent session state** | Redis checkpoint entries per eval sample | `eval-` prefix for identification; future: no-op checkpointer |

---

## Verification Plan

1. **Unit tests**: `uv run pytest -v --tb=short --cov=src --cov-report=term-missing`
2. **Integration test** (requires services running):
   - Ingest a document via `uv run streamlit run main.py`
   - Run evaluation: `uv run python evaluate.py -n 5 -v`
   - Inspect `evaluation_results.json` — scores between 0-1, reasoning present, summary correct
3. **Judge model override**: `JUDGE_MODEL=<stronger-model> uv run python evaluate.py -n 3` — verify different model is used in logs
