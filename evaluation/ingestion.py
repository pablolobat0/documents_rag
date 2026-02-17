import logging

from rich.progress import Progress

from evaluation.schemas import EvalSample
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.value_objects.page_content import PageContent
from src.infrastructure.processing.text_splitter import LangchainTextSplitter

logger = logging.getLogger(__name__)


def ingest_samples(
    samples: list[EvalSample],
    vector_store: VectorStorePort,
    progress: Progress,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> None:
    """Split sample contexts into chunks and upsert into the evaluation collection."""
    splitter = LangchainTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    task = progress.add_task("Ingesting samples...", total=len(samples))

    batch_size = 50
    all_chunks = []

    for sample in samples:
        page = PageContent(
            content=sample.context,
            page_number=sample.section_id,
        )
        base_metadata = {
            "doc_id": sample.doc_id,
            "section_id": sample.section_id,
            "title": sample.title,
        }
        chunks = splitter.split_pages([page], base_metadata)
        all_chunks.extend(chunks)
        progress.advance(task)

    # Upsert in batches
    upsert_task = progress.add_task("Upserting chunks...", total=len(all_chunks))
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vector_store.upsert_chunks(batch)
        progress.advance(upsert_task, advance=len(batch))

    logger.info("Ingested %d chunks from %d samples", len(all_chunks), len(samples))
