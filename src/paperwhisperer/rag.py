"""Assemble the RAG chain: retrieve -> prompt -> LLM -> answer (+ sources).

This is the "G" in RAG. Retrieved chunks are stuffed into the prompt as
context, the LLM is constrained to answer only from that context and to cite
page numbers, and we return both the answer and the source documents.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from paperwhisperer.llm import get_llm
from paperwhisperer.retriever import get_hybrid_retriever

SYSTEM_PROMPT = (
    "You are PaperWhisperer, a precise research assistant. Answer the "
    "question using ONLY the provided context from the user's document.\n\n"
    "Rules:\n"
    "- If the answer is not in the context, reply exactly: "
    '"I could not find that in the document." Do not use outside knowledge.\n'
    "- Cite the page number(s) you used in parentheses, e.g. (page 3).\n"
    "- Be concise and accurate; quote short key phrases when useful."
)

HUMAN_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks into one context string tagged with page numbers.

    The [page N] tags are what let the LLM cite sources accurately.
    """
    return "\n\n".join(
        f"[page {d.metadata.get('page_label', '?')}] {d.page_content}"
        for d in docs
    )


def build_answer_chain():
    """Build the streamable chain: {context: [docs], question} -> answer str.

    This is the generation half (no retrieval). It's streamable, so the UI can
    render tokens live via st.write_stream.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    return (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def build_rag_chain(retriever=None):
    """Build an LCEL chain that maps a question -> {answer, context, question}.

    Defaults to the hybrid (vector + BM25) retriever. Returning the source
    documents alongside the answer lets callers display citations and lets us
    debug what the model actually saw.
    """
    if retriever is None:
        retriever = get_hybrid_retriever()

    # Retrieve once; expose both the source docs and the generated answer.
    return RunnableParallel(
        context=retriever,
        question=RunnablePassthrough(),
    ).assign(answer=build_answer_chain())
