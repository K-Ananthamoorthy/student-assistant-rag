"""Streamlit chat UI. Thin by design, all RAG/agent logic lives in rag/.

Flow: upload gate -> Analyse -> document cards -> chat / literature review.
The chat never appears until at least one document is indexed.
"""

import uuid

import streamlit as st

from rag.agent import ask, build_agent
from rag.ingest import clear_index, get_vectorstore, ingest_pdfs, list_papers

STARTERS = [
    "Summarize the key points across my documents",
    "Compare the methods used in these documents",
    "What gaps do these documents leave open?",
]


@st.cache_resource
def get_graph():
    return build_agent()


def analyse(files):
    with st.status("Analysing your documents...", expanded=True) as status:
        cards = ingest_pdfs(files, on_progress=st.write)
        status.update(label=f"Analysed {len(cards)} document(s)", state="complete")


def upload_gate():
    """Fullscreen first step: nothing else exists until documents are analysed."""
    st.title("🔬 Doc Companion")
    st.subheader("Chat with your PDFs. Fully local and private.")
    st.write(
        "Step 1 of 2: upload your PDFs. Research papers, class notes, "
        "reports, books, personal documents, anything you work with."
    )

    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Analyse documents", type="primary", disabled=not pdfs):
        analyse(pdfs)
        st.rerun()
    st.stop()


def document_cards():
    papers = list_papers()
    with st.expander(f"📄 Your documents ({len(papers)})", expanded=not st.session_state.history):
        for card in papers.values():
            st.markdown(f"**{card['title']}**  \n*{card['topic']}*")
            st.caption(f"Type: {card['method'] or 'not identified'} · {card.get('pages', '?')} pages")
            for finding in card["findings"]:
                st.markdown(f"- {finding}")
            st.divider()


def sidebar():
    with st.sidebar:
        st.header("Your documents")
        for name, card in list_papers().items():
            st.markdown(f"📄 **{name}**")
            st.caption(f"{card.get('pages', '?')} pages · {card['topic'][:60]}")
        st.caption(f"{len(get_vectorstore().get()['ids'])} chunks indexed")

        more = st.file_uploader("Add more PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Analyse", disabled=not more):
            analyse(more)
            st.rerun()

        st.divider()
        if st.button("New conversation", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.history = []
            st.rerun()
        if st.button("Start over (remove documents)", use_container_width=True):
            clear_index()
            st.session_state.clear()
            st.rerun()


def chat_page():
    st.title("🔬 Doc Companion")
    document_cards()

    sidebar()

    if not st.session_state.history:
        st.caption("Try one of these to get started:")
        cols = st.columns(len(STARTERS))
        for col, starter in zip(cols, STARTERS):
            if col.button(starter):
                st.session_state.pending_q = starter

    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    question = st.chat_input("Ask about your documents...") or st.session_state.pop(
        "pending_q", None
    )
    if question:
        st.chat_message("user").write(question)
        st.session_state.history.append(("user", question))
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            answer = ask(get_graph(), question, st.session_state.thread_id)
            st.write(answer)
        st.session_state.history.append(("assistant", answer))


def main():
    st.set_page_config(page_title="Doc Companion | Local RAG", page_icon="🔬")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []

    if len(get_vectorstore().get()["ids"]) == 0:
        upload_gate()
    chat_page()


if __name__ == "__main__":
    main()
