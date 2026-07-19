"""RAG evaluation harness: golden Q&A set scored by an LLM judge.

Measures the two things that matter for RAG quality:
  faithfulness — is the answer grounded in the retrieved context? (no hallucination)
  correctness  — does the answer match the reference answer? (retrieval + generation worked)

Run:  uv run python -m evals.run_eval
"""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config
from rag.guards import yes_no as judge

EVAL_DIR = Path(__file__).parent


def build_eval_index() -> Chroma:
    """Index the eval corpus into its own in-memory collection (never touches user data)."""
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    ).split_documents([Document(page_content=(EVAL_DIR / "study_notes.txt").read_text())])
    vs = Chroma(
        collection_name="eval",
        embedding_function=OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL),
    )
    vs.add_documents(chunks)
    return vs


def main():
    vs = build_eval_index()
    llm = ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0)
    golden = [json.loads(line) for line in (EVAL_DIR / "golden.jsonl").read_text().splitlines()]

    results = []
    for case in golden:
        docs = vs.similarity_search(case["question"], k=config.RETRIEVAL_K)
        context = "\n\n".join(d.page_content for d in docs)
        answer = llm.invoke(
            "Answer the question using ONLY the context below.\n\n"
            f"Context:\n{context}\n\nQuestion: {case['question']}"
        ).content

        faithful = judge(
            f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
            "Is the answer supported by the context? Ignore real-world truth — "
            "only compare it to the context.\nReply with exactly one word: YES or NO."
        )
        correct = judge(
            f"Reference answer:\n{case['reference']}\n\nCandidate answer:\n{answer}\n\n"
            "Does the candidate convey the same key facts as the reference?\n"
            "Reply with exactly one word: YES or NO."
        )
        results.append((case["question"], faithful, correct))
        print(f"{'PASS' if faithful and correct else 'FAIL'}  "
              f"faithful={faithful} correct={correct}  {case['question']}")

    n = len(results)
    print(f"\nfaithfulness: {sum(f for _, f, _ in results)}/{n}")
    print(f"correctness:  {sum(c for _, _, c in results)}/{n}")


if __name__ == "__main__":
    main()
