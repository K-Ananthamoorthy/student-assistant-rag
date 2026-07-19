"""Safety guardrails: the answer must be supported by the retrieved context.

RAG's whole value is grounding — if the model says something the documents
don't support, we'd rather refuse than hallucinate.

Judge design note: with a 3B local model, JSON-constrained structured output
proved unreliable for verdicts (tested: it defaulted to false/unsupported
regardless of input). A plain-text YES/NO answer is what the small model can
do reliably, so that's what we use.
# ponytail: 3B judge catches gross hallucination but misses subtle mixed
# claims; swap CHAT_MODEL for an 8B+ judge when hardware allows.
"""

import re

from langchain_ollama import ChatOllama

from rag import config

REFUSAL = (
    "I can't answer that from the uploaded documents. "
    "Try rephrasing, or upload a document that covers this topic."
)

_JUDGE_PROMPT = """Context:
{context}

Answer:
{answer}

Question: Is the answer above supported by the context above? The answer is
supported if its claims appear in the context. Ignore whether the answer is
true in the real world — only compare it to the context.

Reply with exactly one word: YES or NO."""


def yes_no(prompt: str) -> bool:
    """Ask the local model a YES/NO question; fail open on errors so a judge
    outage never takes the app down."""
    llm = ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0)
    try:
        return llm.invoke(prompt).content.strip().upper().startswith("YES")
    except Exception:
        return True


def is_grounded(answer: str, context: str) -> bool:
    return yes_no(_JUDGE_PROMPT.format(context=context, answer=answer))


_TAG = re.compile(r"\[([^\[\]]+? p\.\d+)\]")


def fix_citations(answer: str, context: str) -> str:
    """Deterministic citation guard: the 3B model sometimes invents page
    numbers (writes p.3 for a chunk tagged p.1). Every cited tag must exist
    verbatim in the retrieved context; an invented tag is replaced with the
    real tag of the same source file, or dropped if none matches."""
    valid = set(_TAG.findall(context))

    def repair(match: re.Match) -> str:
        tag = match.group(1)
        if tag in valid:
            return match.group(0)
        source = tag.rsplit(" p.", 1)[0]
        for v in sorted(valid):
            if v.rsplit(" p.", 1)[0] == source:
                return f"[{v}]"
        return ""

    return _TAG.sub(repair, answer).strip()
