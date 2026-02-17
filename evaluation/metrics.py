import logging
import math
import statistics
import time
from collections.abc import Callable

from rich.progress import Progress

from evaluation.schemas import EvalSample, RetrievalMetrics, RetrievalResult
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.value_objects.retrieved_document import RetrievedDocument

logger = logging.getLogger(__name__)


def evaluate_retrieval(
    samples: list[EvalSample],
    search_fn: Callable[[str, int], list[RetrievedDocument]],
    top_k: int,
    vector_store: VectorStorePort,
    progress: Progress,
) -> tuple[list[RetrievalResult], RetrievalMetrics]:
    """Run retrieval evaluation over all samples and compute aggregate metrics."""
    task = progress.add_task("Evaluating retrieval...", total=len(samples))

    # Cache total_relevant counts per (doc_id, section_id)
    relevant_counts: dict[tuple[str, int], int] = {}

    results: list[RetrievalResult] = []

    for sample in samples:
        start = time.perf_counter()
        search_results = search_fn(sample.question, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Find positions where both doc_id and section_id match
        relevant_positions = [
            i
            for i, doc in enumerate(search_results)
            if doc.metadata
            and doc.metadata.get("doc_id") == sample.doc_id
            and doc.metadata.get("section_id") == sample.section_id
        ]

        # Count total relevant chunks for this (doc_id, section_id) (cached)
        cache_key = (sample.doc_id, sample.section_id)
        if cache_key not in relevant_counts:
            relevant_counts[cache_key] = vector_store.count_chunks(
                {"doc_id": sample.doc_id, "section_id": sample.section_id}
            )

        results.append(
            RetrievalResult(
                question=sample.question,
                doc_id=sample.doc_id,
                relevant_positions=relevant_positions,
                total_relevant=relevant_counts[cache_key],
                latency_ms=elapsed_ms,
            )
        )
        progress.advance(task)

    metrics = _compute_metrics(results)
    return results, metrics


def _compute_metrics(results: list[RetrievalResult]) -> RetrievalMetrics:
    """Aggregate per-query metrics into overall RetrievalMetrics."""
    if not results:
        return RetrievalMetrics()

    recall_values = [_recall_at_k(r) for r in results]
    mrr_values = [_mrr(r) for r in results]
    ndcg5_values = [_ndcg_at_k(r, 5) for r in results]
    ndcg10_values = [_ndcg_at_k(r, 10) for r in results]
    latencies = [r.latency_ms for r in results]

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return RetrievalMetrics(
        recall_at_25=statistics.mean(recall_values),
        mrr=statistics.mean(mrr_values),
        ndcg_at_5=statistics.mean(ndcg5_values),
        ndcg_at_10=statistics.mean(ndcg10_values),
        mean_latency_ms=statistics.mean(latencies),
        p50_latency_ms=sorted_latencies[n // 2],
        p95_latency_ms=sorted_latencies[int(n * 0.95)],
        total_queries=n,
    )


def _recall_at_k(result: RetrievalResult) -> float:
    """Compute recall: fraction of relevant docs found in top-K."""
    if result.total_relevant == 0:
        return 0.0
    return len(result.relevant_positions) / result.total_relevant


def _mrr(result: RetrievalResult) -> float:
    """Compute MRR: reciprocal rank of first relevant result."""
    if not result.relevant_positions:
        return 0.0
    return 1.0 / (min(result.relevant_positions) + 1)


def _ndcg_at_k(result: RetrievalResult, k: int) -> float:
    """Compute NDCG@K with binary relevance."""
    if result.total_relevant == 0:
        return 0.0

    # DCG: sum 1/log2(i+2) for relevant positions within top-K
    dcg = sum(1.0 / math.log2(pos + 2) for pos in result.relevant_positions if pos < k)

    # Ideal DCG: best possible with total_relevant items in top positions
    ideal_count = min(result.total_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0:
        return 0.0
    return dcg / idcg
