"""Chat model factory -- the LLM side of the provider-agnostic seam.

Like embeddings.py, this is the single place that names the vendor. Swapping
Gemini for another chat model later means editing only this file.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from paperwhisperer.config import settings


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Return the configured Gemini chat model.

    temperature defaults to 0.0 for factual, grounded, reproducible answers
    (RAG wants faithfulness to the source, not creativity).
    """
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key.get_secret_value(),
        temperature=temperature,
        # gRPC uses its own TLS stack that truststore does not patch; REST
        # routes through httpx and verifies via the OS trust store.
        transport="rest",
    )
