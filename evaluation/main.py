import json
import logging
import statistics
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from evaluation.container import EvalContainer
from evaluation.dataset import load_eval_samples
from evaluation.ingestion import ingest_samples
from evaluation.metrics import evaluate_retrieval
from evaluation.schemas import EvalSample, RetrievalMetrics, RetrievalResult
from evaluation.settings import EvalSettings, Mode
from src.domain.value_objects.chat_message import ChatMessage

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="RAG pipeline evaluation tool.")


@app.command()
def evaluate(
    mode: Mode = typer.Option(Mode.full, help="Evaluation mode"),
    top_k: int = typer.Option(100, help="Retrieval top-K"),
    output: Path | None = typer.Option(None, help="JSON results output path"),
    force_reingest: bool = typer.Option(
        False, "--force-reingest", help="Recreate evaluation collection"
    ),
    verbose: bool = typer.Option(False, help="Enable debug logging"),
):
    """Evaluate the RAG pipeline using the RAGBench dataset."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Disable login to clean tool ouput
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    eval_settings = EvalSettings()

    all_results: dict = {"mode": mode.value}
    retrieval_data: tuple[list[RetrievalResult], RetrievalMetrics] | None = None
    generation_data: list[dict] | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        # Load eval samples
        load_task = progress.add_task("Loading eval samples...", total=None)
        samples = load_eval_samples()
        progress.update(load_task, completed=True, total=1)
        console.print(f"Loaded [bold]{len(samples)}[/bold] evaluation samples")

        # Check collection and ingest if needed
        container = EvalContainer(eval_settings)
        needs_ingest = force_reingest or not container.vector_store.collection_exists()

        if needs_ingest:
            if force_reingest and container.vector_store.collection_exists():
                container.vector_store.delete_collection()
                console.print("Deleted existing evaluation collection")
                # Recreate collection
                container = EvalContainer(eval_settings)

            ingest_samples(
                samples=samples,
                vector_store=container.vector_store,
                progress=progress,
                chunk_size=eval_settings.chunk_size,
                chunk_overlap=eval_settings.chunk_overlap,
            )
            console.print("[green]Ingestion complete[/green]")
        else:
            console.print("Evaluation collection exists, skipping ingestion")

        if mode in (Mode.retrieval, Mode.full):
            retrieval_data = evaluate_retrieval(
                samples=samples,
                search_fn=container.vector_store.search,
                top_k=top_k,
                vector_store=container.vector_store,
                progress=progress,
            )

        if mode in (Mode.generation, Mode.full):
            generation_data = _collect_generation_results(container, samples, progress)

    # Display results after progress bars are cleared
    if retrieval_data is not None:
        results, metrics = retrieval_data
        _display_retrieval_results(results, metrics, verbose)
        all_results["retrieval"] = {
            "recall_at_25": metrics.recall_at_25,
            "mrr": metrics.mrr,
            "ndcg_at_5": metrics.ndcg_at_5,
            "ndcg_at_10": metrics.ndcg_at_10,
            "mean_latency_ms": metrics.mean_latency_ms,
            "p50_latency_ms": metrics.p50_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "total_queries": metrics.total_queries,
        }

    if generation_data is not None:
        all_results["generation"] = _display_generation_results(generation_data)

    if output:
        with open(output, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\nResults saved to [bold]{output}[/bold]")


def _collect_generation_results(
    container: EvalContainer,
    samples: list[EvalSample],
    progress: Progress,
) -> list[dict]:
    """Run generation evaluation and return per-sample results."""
    task = progress.add_task("Evaluating generation...", total=len(samples))
    num_docs = container._settings.retrieval_num_documents
    results = []

    for i, sample in enumerate(samples):
        messages = [ChatMessage(role="user", content=sample.question)]
        answer = container.agent.run(messages, session_id=f"eval-{i}")

        search_results = container.vector_store.search(sample.question, num_docs)
        contexts = [doc.page_content for doc in search_results]

        faithfulness = container.judge.evaluate_faithfulness(
            sample.question, contexts, answer
        )
        relevance = container.judge.evaluate_answer_relevance(sample.question, answer)

        results.append(
            {
                "question": sample.question,
                "answer": answer,
                "faithfulness_score": faithfulness.score,
                "faithfulness_reasoning": faithfulness.reasoning,
                "relevance_score": relevance.score,
                "relevance_reasoning": relevance.reasoning,
            }
        )
        progress.advance(task)

    return results


def _display_retrieval_results(
    results: list[RetrievalResult],
    metrics: RetrievalMetrics,
    verbose: bool,
) -> None:
    """Display retrieval results table and metrics panel."""
    if verbose:
        table = Table(title="Retrieval Results")
        table.add_column("Question", max_width=50)
        table.add_column("Doc ID", max_width=20)
        table.add_column("Hits", justify="center")
        table.add_column("Latency (ms)", justify="right")

        for r in results:
            hits = len(r.relevant_positions)
            hit_str = f"[green]{hits}[/green]" if hits > 0 else "[red]0[/red]"
            table.add_row(
                r.question[:50],
                r.doc_id[:20],
                hit_str,
                f"{r.latency_ms:.0f}",
            )

        console.print(table)

    summary = (
        f"[bold]Recall@25:[/bold]     {metrics.recall_at_25:.4f}\n"
        f"[bold]MRR:[/bold]           {metrics.mrr:.4f}\n"
        f"[bold]NDCG@5:[/bold]        {metrics.ndcg_at_5:.4f}\n"
        f"[bold]NDCG@10:[/bold]       {metrics.ndcg_at_10:.4f}\n"
        f"\n"
        f"[bold]Latency mean:[/bold]  {metrics.mean_latency_ms:.0f} ms\n"
        f"[bold]Latency p50:[/bold]   {metrics.p50_latency_ms:.0f} ms\n"
        f"[bold]Latency p95:[/bold]   {metrics.p95_latency_ms:.0f} ms\n"
        f"[bold]Total queries:[/bold] {metrics.total_queries}"
    )
    console.print(Panel(summary, title="Retrieval Metrics"))


def _display_generation_results(results: list[dict]) -> dict:
    """Display generation results table and summary, return metrics dict."""
    faith_scores = [r["faithfulness_score"] for r in results]
    rel_scores = [r["relevance_score"] for r in results]

    table = Table(title="Generation Results")
    table.add_column("Question", max_width=40)
    table.add_column("Faith.", justify="center", width=6)
    table.add_column("Relev.", justify="center", width=6)
    table.add_column("Reasoning", max_width=50)

    for r in results:
        reasoning = r["faithfulness_reasoning"][:50]
        table.add_row(
            r["question"][:40],
            str(r["faithfulness_score"]),
            str(r["relevance_score"]),
            reasoning,
        )

    console.print(table)

    summary = (
        f"[bold]Faithfulness:[/bold] {_format_stats(faith_scores)}\n"
        f"[bold]Relevance:[/bold]   {_format_stats(rel_scores)}"
    )
    console.print(Panel(summary, title="Generation Summary"))

    return {
        "faithfulness": _format_stats(faith_scores),
        "relevance": _format_stats(rel_scores),
        "details": results,
    }


def _format_stats(scores: list[int]) -> str:
    """Format descriptive statistics for a list of scores."""
    if not scores:
        return "No data"
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return (
        f"Mean: {mean:.2f}  Median: {median:.1f}  "
        f"Min: {min(scores)}  Max: {max(scores)}  Std: {std:.2f}"
    )


if __name__ == "__main__":
    app()
