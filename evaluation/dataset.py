import json
import logging

from datasets import load_dataset

from evaluation.schemas import EvalSample
from evaluation.settings import CACHE_PATH, DATASET_NAME, DATASET_SPLIT

logger = logging.getLogger(__name__)


def load_eval_samples(n: int = 200, seed: int = 42) -> list[EvalSample]:
    """Load evaluation samples from cache or HuggingFace dataset.

    Downloads the dataset, samples `n` rows with a fixed seed,
    and caches results for reproducibility.
    """
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            data = json.load(f)
        return [EvalSample(**row) for row in data]

    logger.info("Downloading dataset %s...", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    sampled = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    samples = [
        EvalSample(
            question=row["question"],
            doc_id=row["doc_id"],
            section_id=row["section_id"],
            context=row["context"],
            title=row["title"],
        )
        for row in sampled
    ]

    # Cache for reproducibility
    with open(CACHE_PATH, "w") as f:
        json.dump(
            [
                {
                    "question": s.question,
                    "doc_id": s.doc_id,
                    "section_id": s.section_id,
                    "context": s.context,
                    "title": s.title,
                }
                for s in samples
            ],
            f,
            indent=2,
        )
    logger.info("Cached %d samples to %s", len(samples), CACHE_PATH)

    return samples
