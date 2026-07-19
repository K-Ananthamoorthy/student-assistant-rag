"""Agentic RAG as a LangGraph state machine.

The LLM is the reasoning engine: it DECIDES whether to retrieve, GRADES what
came back, REWRITES the question when retrieval was poor (self-correction),
and only then answers — grounded, with citations, behind a guardrail.

    START -> route --(small talk)-----> chitchat -> END
               |   --(corpus-level)---> overview -> END   (answers from paper cards)
               |
               v (specific question)
             agent --(no tool call)--> END
               |
               v (tool call)
            retrieve --> grade --(relevant)--> generate -> guard -> END
                           |
                           v (irrelevant, < MAX_REWRITES)
                        rewrite -> agent   (loop)

Routing notes:
- Corpus-level questions ("summarize all my papers", "compare them") are a
  known weakness of top-k retrieval: no k chunks represent the whole corpus.
  They route to an overview node that answers from the per-paper cards built
  at ingestion time instead of from retrieval.
- Ideally the agent itself decides when NOT to retrieve, but llama3.2:3b
  calls the tool on every message once tools are bound (measured; prompting
  doesn't change it). So routing is an explicit few-shot classifier node;
  with a larger model it could be deleted in favor of agent discretion.
"""

import sqlite3

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag import config, guards
from rag.ingest import get_vectorstore, list_papers


class AgentState(MessagesState):
    rewrites: int


def _llm(temperature: float = 0) -> ChatOllama:
    return ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=temperature)


def _last_question(state: AgentState) -> str:
    return next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


def _last_tool_content(state: AgentState) -> str:
    return next(m.content for m in reversed(state["messages"]) if m.type == "tool")


def build_agent():
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": config.RETRIEVAL_K})

    @tool
    def retrieve(query: str) -> str:
        """Search the uploaded PDF documents for passages relevant to the query."""
        docs = retriever.invoke(query)
        if not docs:
            return "No documents indexed yet."
        return "\n\n".join(
            f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )

    llm_with_tools = _llm().bind_tools([retrieve])

    # --- nodes -------------------------------------------------------------

    def route(state: AgentState) -> str:
        """Three-way routing: small talk / corpus-level / specific question.

        Few-shot examples are load-bearing: without them the 3B model sends
        comparison and gap questions to SEARCH (tested at 6/8, 8/8 with them).
        """
        prompt = (
            "A student is using a research assistant loaded with their project papers.\n"
            "Classify their message:\n"
            "- CHAT: greeting, thanks, or small talk\n"
            "- OVERVIEW: asks about the paper collection as a whole: summaries of all "
            "papers, comparisons between papers, gaps or open questions, what the "
            "collection covers\n"
            "- SEARCH: asks for specific information found inside the papers\n\n"
            "Examples:\n"
            '"hi there" -> CHAT\n'
            '"Summarize my papers" -> OVERVIEW\n'
            '"Compare the approaches in these papers" -> OVERVIEW\n'
            '"What open problems remain across these papers?" -> OVERVIEW\n'
            '"What dataset did the authors use?" -> SEARCH\n'
            '"Define precision and recall" -> SEARCH\n\n'
            f'Message: "{_last_question(state)}"\n\n'
            "Reply with exactly one word: CHAT, OVERVIEW, or SEARCH."
        )
        try:
            verdict = _llm().invoke(prompt).content.strip().upper().rstrip(".")
        except Exception:
            verdict = "SEARCH"
        return {"CHAT": "chitchat", "OVERVIEW": "overview"}.get(verdict, "agent")

    def overview(state: AgentState):
        """Corpus-level answers come from the paper cards, not top-k retrieval."""
        papers = list_papers()
        cards = "\n\n".join(
            f"Paper: {c['title']}\nTopic: {c['topic']}\nMethod: {c['method']}\n"
            + "\n".join(f"- {p}" for p in c["findings"])
            for c in papers.values()
        )
        answer = _llm().invoke(
            "A student asks about their collection of project papers. Answer from "
            "the paper summaries below. Name papers by title. Do not invent papers. "
            "Do not use the em dash character.\n\n"
            f"Paper summaries:\n{cards}\n\nQuestion: {_last_question(state)}"
        )
        return {"messages": [answer]}

    def chitchat(state: AgentState):
        """Direct reply, no tools bound — the model can't wrongly retrieve here."""
        response = _llm(temperature=0.4).invoke(
            [
                {
                    "role": "system",
                    "content": "You are a friendly research companion helping a student with "
                    "their project papers. Reply briefly and warmly; invite them to ask about "
                    "their papers. Do not use the em dash character.",
                },
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    def agent(state: AgentState):
        """Reasoning step: answer directly, or emit a retrieve tool call."""
        system = (
            "You are a research assistant for the user's uploaded project papers. "
            "For any question about paper content, call the retrieve tool. "
            "Only answer directly for greetings or questions about the conversation itself."
        )
        response = llm_with_tools.invoke(
            [{"role": "system", "content": system}, *state["messages"]]
        )
        return {"messages": [response]}

    def grade(state: AgentState) -> str:
        """Conditional edge: are the retrieved chunks actually relevant?"""
        if state.get("rewrites", 0) >= config.MAX_REWRITES:
            return "generate"  # stop looping; answer with what we have
        # Plain-text YES/NO: reliable for a 3B model where JSON-forced
        # verdicts are not (see rag/guards.py judge design note).
        relevant = guards.yes_no(
            f"Question: {_last_question(state)}\n\n"
            f"Retrieved passages:\n{_last_tool_content(state)}\n\n"
            "Do these passages contain information that can answer the question?\n"
            "Reply with exactly one word: YES or NO."
        )
        return "generate" if relevant else "rewrite"

    def rewrite(state: AgentState):
        """Self-correction: rephrase the question for better retrieval."""
        better = _llm(temperature=0.4).invoke(
            "Rewrite this question to retrieve better search results from a "
            f"document index. Reply with ONLY the rewritten question.\n\n{_last_question(state)}"
        )
        return {
            "messages": [HumanMessage(content=better.content)],
            "rewrites": state.get("rewrites", 0) + 1,
        }

    def generate(state: AgentState):
        """Grounded answer with citations, checked by the grounding guard."""
        context = _last_tool_content(state)
        answer = _llm().invoke(
            "You are answering a research question from a student's project papers. "
            "Answer using ONLY the context below. Synthesize across papers when "
            "several are relevant, and note when sources disagree. "
            "Cite sources inline by copying the exact [source p.N] tags that "
            "precede the passages you used, never an invented tag. "
            "If the context doesn't contain the answer, say so. "
            "Do not use the em dash character.\n\n"
            f"Context:\n{context}\n\nQuestion: {_last_question(state)}"
        )
        if guards.is_grounded(answer.content, context):
            answer.content = guards.fix_citations(answer.content, context)
        else:
            answer.content = guards.REFUSAL
        return {"messages": [answer], "rewrites": 0}

    # --- graph -------------------------------------------------------------

    g = StateGraph(AgentState)
    g.add_node("chitchat", chitchat)
    g.add_node("overview", overview)
    g.add_node("agent", agent)
    g.add_node("retrieve", ToolNode([retrieve]))
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.add_conditional_edges(
        START, route, {"chitchat": "chitchat", "overview": "overview", "agent": "agent"}
    )
    g.add_edge("chitchat", END)
    g.add_edge("overview", END)
    # tools_condition routes: tool call -> "retrieve", plain answer -> END
    g.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    g.add_conditional_edges("retrieve", grade, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "agent")
    g.add_edge("generate", END)

    # SQLite checkpointer = conversation state persists across app restarts.
    conn = sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False)
    return g.compile(checkpointer=SqliteSaver(conn))


def ask(graph, question: str, thread_id: str = "default") -> str:
    """One turn of conversation on a persistent thread."""
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content
