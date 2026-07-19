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
