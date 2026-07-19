"""Streamlit chat UI. Thin by design, all RAG/agent logic lives in rag/.

Flow: upload gate -> Analyse -> paper cards -> chat / literature review.
The chat never appears until at least one paper is indexed.
"""

import uuid

import streamlit as st

from rag.agent import ask, build_agent
from rag.ingest import (
    clear_index,
    get_vectorstore,
    ingest_pdfs,
    list_papers,
    literature_review,
)

STARTERS = [
    "Summarize the key findings across my papers",
    "Compare the methods used in these papers",
    "What gaps do these papers leave open?",
]


@st.cache_resource
def get_graph():
    return build_agent()


def analyse(files):
    with st.status("Analysing your papers...", expanded=True) as status:
        cards = ingest_pdfs(files, on_progress=st.write)
        status.update(label=f"Analysed {len(cards)} paper(s)", state="complete")


def upload_gate():
    """Fullscreen first step: nothing else exists until papers are analysed."""
    st.title("🔬 Research Companion")
    st.subheader("Your project papers, understood. Fully local and private.")
    st.write(
        "Step 1 of 2: upload the PDFs for your research project. "
        "Papers, reports, notes, anything you are working from."
    )

    pdfs = st.file_uploader("Upload papers (PDF)", type="pdf", accept_multiple_files=True)
    if st.button("Analyse papers", type="primary", disabled=not pdfs):
        analyse(pdfs)
        st.rerun()
    st.stop()


def paper_cards():
    papers = list_papers()
    with st.expander(f"📄 Your papers ({len(papers)})", expanded=not st.session_state.history):
        for card in papers.values():
            st.markdown(f"**{card['title']}**  \n*{card['topic']}*")
            st.caption(f"Method: {card['method'] or 'not identified'} · {card.get('pages', '?')} pages")
            for finding in card["findings"]:
                st.markdown(f"- {finding}")
            st.divider()


def sidebar():
    with st.sidebar:
        st.header("Project")
        st.caption(f"{len(get_vectorstore().get()['ids'])} chunks indexed")

        more = st.file_uploader("Add more papers", type="pdf", accept_multiple_files=True)
        if st.button("Analyse", disabled=not more):
            analyse(more)
            st.rerun()

        st.divider()
        if st.button("📝 Draft literature review", use_container_width=True):
            with st.spinner("Drafting from your paper summaries..."):
                st.session_state.review = literature_review()

        st.divider()
        if st.button("New conversation", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.history = []
            st.rerun()
        if st.button("Start over (new project)", use_container_width=True):
            clear_index()
            st.session_state.clear()
            st.rerun()


def chat_page():
    st.title("🔬 Research Companion")
    paper_cards()

    if st.session_state.get("review"):
        with st.expander("📝 Literature review draft", expanded=True):
            st.markdown(st.session_state.review)
            st.download_button(
                "Download as markdown", st.session_state.review, "literature_review.md"
            )

    sidebar()

    if not st.session_state.history:
        st.caption("Try one of these to get started:")
        cols = st.columns(len(STARTERS))
        for col, starter in zip(cols, STARTERS):
            if col.button(starter):
                st.session_state.pending_q = starter

    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    question = st.chat_input("Ask across your papers...") or st.session_state.pop(
        "pending_q", None
    )
    if question:
        st.chat_message("user").write(question)
        st.session_state.history.append(("user", question))
        with st.chat_message("assistant"), st.spinner("Researching..."):
            answer = ask(get_graph(), question, st.session_state.thread_id)
            st.write(answer)
        st.session_state.history.append(("assistant", answer))


def main():
    st.set_page_config(page_title="Research Companion | Local RAG", page_icon="🔬")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []

    if len(get_vectorstore().get()["ids"]) == 0:
        upload_gate()
    chat_page()


if __name__ == "__main__":
    main()
