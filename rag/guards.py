"""Grounding guardrails: check that an answer is supported by its context."""

import re

from langchain_ollama import ChatOllama

from rag import config

REFUSAL = (
    "I couldn't find anything about that in your documents. "
    "Try rephrasing, or upload a document that covers this topic."
)

# Appended to an answer the grounding judge could not verify.
CAUTION = (
    "\n\n*Note: I could not fully verify every claim above against your "
    "documents. Please double-check the cited pages.*"
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
    """Ask the model a YES/NO question. Fails open on error."""
    llm = ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0)
    try:
        return llm.invoke(prompt).content.strip().upper().startswith("YES")
    except Exception:
        return True


def is_grounded(answer: str, context: str) -> bool:
    return yes_no(_JUDGE_PROMPT.format(context=context, answer=answer))


_TAG = re.compile(r"\[([^\[\]]+? p\.\d+)\]")


def fix_citations(answer: str, context: str) -> str:
    """Keep only [source p.N] tags present in the context. Remap an invented
    tag to a real one from the same source, or drop it if none matches."""
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
