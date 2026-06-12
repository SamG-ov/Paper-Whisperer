"""PaperWhisperer -- Streamlit chat UI.

Run with:  streamlit run app.py

Streamlit re-runs this whole script on every interaction, so:
  - persistent state (chat history) lives in st.session_state
  - expensive resources (vector store, chain) are memoized with cache_resource
"""

import tempfile
from pathlib import Path

import streamlit as st

from paperwhisperer.ingestion.chunker import split_documents
from paperwhisperer.ingestion.loader import load_pdf
from paperwhisperer.rag import build_answer_chain
from paperwhisperer.vector_store import get_vector_store, index_documents

st.set_page_config(page_title="PaperWhisperer", page_icon="📄")


# --- Cached resources: built once, reused across re-runs ---
@st.cache_resource
def get_store():
    """One shared ChromaDB client for the whole app."""
    return get_vector_store()


@st.cache_resource
def get_answer_chain():
    """The streamable generation chain (built once)."""
    return build_answer_chain()


def unique_pages(docs) -> str:
    """Comma-separated, numerically sorted list of source page labels."""
    def key(p: str) -> int:
        return int(p) if p.isdigit() else 9999

    labels = {d.metadata.get("page_label", "?") for d in docs}
    return ", ".join(sorted(labels, key=key))


# --- Sidebar: upload & index a PDF ---
with st.sidebar:
    st.header("📚 Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("Index document", use_container_width=True):
        with st.spinner(f"Indexing {uploaded.name} …"):
            tmp = Path(tempfile.gettempdir()) / uploaded.name
            tmp.write_bytes(uploaded.getvalue())
            chunks = split_documents(load_pdf(tmp))
            index_documents(chunks, store=get_store())
        st.success(f"Indexed {uploaded.name} — {len(chunks)} chunks.")

    st.caption(f"Vector store: {get_store()._collection.count()} chunks indexed.")


# --- Main: chat ---
st.title("📄 PaperWhisperer")
st.caption("Chat with your PDFs — grounded answers with page citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far (it lives in session_state across re-runs).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📑 Sources: page(s) {msg['sources']}")

# Handle a new question.
if question := st.chat_input("Ask a question about the document…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Retrieve from the shared cached store (single client).
        retriever = get_store().as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20}
        )
        docs = retriever.invoke(question)
        # Stream the answer token-by-token; write_stream returns the full text.
        answer = st.write_stream(
            get_answer_chain().stream({"context": docs, "question": question})
        )
        sources = unique_pages(docs)
        st.caption(f"📑 Sources: page(s) {sources}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
