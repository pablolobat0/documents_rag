class MetadataClassifierPrompts:
    """Prompts for document metadata classification."""

    CLASSIFICATION_SYSTEM_PROMPT = """You are a document classifier. Given the content of a document, classify it by type and tags.

Available types: "book", "recipe", "project", "prompt", "concept"
Available tags: "AI", "LLM", "investment", "attention", "rag", "transformers", "psychology"

Rules:
- Set type to the single best match, or null if uncertain
- Set tags to a list of matching tags, or null if none apply
- Only assign values you are confident about"""
