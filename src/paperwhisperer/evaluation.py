"""LLM-as-judge evaluation of the RAG pipeline.

We score three dimensions per question -- the same ideas frameworks like
RAGAS measure, implemented transparently with our own Gemini judge:

  * faithfulness -- is every claim in the answer supported by the retrieved
                    context? (catches hallucination)
  * relevancy    -- does the answer actually address the question?
  * correctness  -- does the answer match the reference (ground truth)?

Scores are 1-5 (5 = best). Calls are wrapped with retry so free-tier rate
limits (429s) don't abort a run.
"""

from __future__ import annotations

import time

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from paperwhisperer.llm import get_llm
from paperwhisperer.rag import build_rag_chain, format_docs


class Judgment(BaseModel):
    """A judge's structured verdict (forces a number, not prose)."""

    score: int = Field(description="Integer rating from 1 (worst) to 5 (best).")
    reason: str = Field(description="One short sentence explaining the score.")


_JUDGE_SYSTEM = (
    "You are a strict evaluator of a question-answering system. "
    "Return an integer score from 1 to 5 and a one-sentence reason."
)

FAITHFULNESS = (
    "Rate how well every claim in the ANSWER is supported by the CONTEXT. "
    "5 = fully supported by the context, 1 = mostly unsupported/fabricated.\n\n"
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}"
)
RELEVANCY = (
    "Rate how directly the ANSWER addresses the QUESTION. "
    "5 = perfectly on point, 1 = off-topic.\n\n"
    "QUESTION:\n{question}\n\nANSWER:\n{answer}"
)
CORRECTNESS = (
    "Rate how well the ANSWER matches the REFERENCE in factual content. "
    "If the REFERENCE says the topic is not in the document, then an answer "
    "that declines to answer should score 5.\n\n"
    "QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}"
)


def _judge():
    """A low-temperature, structured-output judge with retry on rate limits."""
    llm = get_llm(temperature=0).with_structured_output(Judgment)
    return llm.with_retry(stop_after_attempt=4, wait_exponential_jitter=True)


def _score(template: str, **kwargs: str) -> Judgment:
    prompt = ChatPromptTemplate.from_messages(
        [("system", _JUDGE_SYSTEM), ("human", template)]
    )
    return (prompt | _judge()).invoke(kwargs)


def evaluate_question(item: dict, rag_chain) -> dict:
    """Run the RAG chain on one question and score the result."""
    question = item["question"]
    reference = item["ground_truth"]

    start = time.perf_counter()
    result = rag_chain.invoke(question)
    latency = time.perf_counter() - start

    answer = result["answer"]
    context = format_docs(result["context"])

    faith = _score(FAITHFULNESS, context=context, answer=answer)
    rel = _score(RELEVANCY, question=question, answer=answer)
    corr = _score(CORRECTNESS, question=question, reference=reference, answer=answer)

    return {
        "question": question,
        "answer": answer,
        "faithfulness": faith.score,
        "relevancy": rel.score,
        "correctness": corr.score,
        "latency_s": round(latency, 2),
    }


def evaluate_dataset(dataset: list[dict]) -> list[dict]:
    """Evaluate every item, reusing one (retry-wrapped) RAG chain."""
    rag_chain = build_rag_chain().with_retry(
        stop_after_attempt=4, wait_exponential_jitter=True
    )
    return [evaluate_question(item, rag_chain) for item in dataset]
