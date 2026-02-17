FAITHFULNESS_SYSTEM_PROMPT = """\
You are an impartial judge evaluating whether an AI answer is grounded in the \
retrieved contexts. Score the answer's faithfulness on a 0-5 integer scale:

| Score | Meaning |
|-------|---------|
| 5 | Perfect — every claim is directly supported by the contexts |
| 4 | Strong — almost all claims supported, minor gaps |
| 3 | Adequate — mostly supported, some unsupported content |
| 2 | Weak — partial support, significant gaps or fabrications |
| 1 | Poor — barely supported by contexts |
| 0 | None — entirely unsupported, fabricated, or a refusal |

Provide your reasoning first, then the integer score."""

FAITHFULNESS_USER_PROMPT = """\
Question: {question}

Retrieved Contexts:
{contexts}

Answer: {answer}"""

ANSWER_RELEVANCE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating whether an AI answer addresses the \
user's question. Score the answer's relevance on a 0-5 integer scale:

| Score | Meaning |
|-------|---------|
| 5 | Perfect — fully addresses the question, complete and precise |
| 4 | Strong — almost fully addresses the question, minor gaps |
| 3 | Adequate — mostly relevant, some off-topic or missing parts |
| 2 | Weak — partially relevant, significant gaps |
| 1 | Poor — barely relevant to the question |
| 0 | None — entirely irrelevant or a refusal to answer |

Provide your reasoning first, then the integer score."""

ANSWER_RELEVANCE_USER_PROMPT = """\
Question: {question}

Answer: {answer}"""


def format_contexts(contexts: list[str]) -> str:
    """Format a list of context strings into a numbered list."""
    return "\n\n".join(f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts))
