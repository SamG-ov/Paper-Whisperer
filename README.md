# 📄 PaperWhisperer

> Chat with your PDFs. Upload research papers, manuals, or documentation and ask questions — get answers grounded in the source text, with citations.

PaperWhisperer is a **Retrieval-Augmented Generation (RAG)** application built to demonstrate
production-minded AI engineering: clean ingestion pipelines, vector search, grounded generation,
evaluation, and containerized deployment.

## 🧰 Tech Stack

| Layer            | Choice                                            |
| ---------------- | ------------------------------------------------- |
| Language         | Python 3.12                                       |
| Orchestration    | LangChain                                          |
| LLM + Embeddings | Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`) |
| Vector Database  | ChromaDB                                           |
| UI               | Streamlit                                          |
| Packaging        | Docker                                             |

## 🚧 Status

Under active development — built milestone by milestone. See the roadmap below.

## 🗺️ Roadmap

- [x] **M0** — Project foundations & setup
- [x] **M1** — PDF ingestion (load & extract text)
- [x] **M2** — Chunking
- [x] **M3** — Embeddings + vector store
- [x] **M4** — Retrieval
- [x] **M5** — Generation with citations
- [x] **M6** — Streamlit UI
- [x] **M7** — Evaluation & observability
- [x] **M8** — Advanced retrieval (hybrid search, caching)
- [ ] **M9** — Dockerize & deploy

## ⚙️ Setup

**Prerequisites:** Python 3.12 (the ML wheel ecosystem lags newer versions) and a
free [Google Gemini API key](https://aistudio.google.com/apikey).

```powershell
# 1. Create & activate a virtual environment (Python 3.12)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies and the package
pip install -r requirements.txt
pip install -e .

# 3. Configure secrets
copy .env.example .env
# then edit .env and add your GOOGLE_API_KEY
```

## ▶️ Usage

```powershell
# Index a PDF, then ask grounded questions about it
python scripts/index_pdf.py path/to/your.pdf
python scripts/ask.py "What problem does this paper solve?"
```

> ⚠️ This project uses Gemini's free tier for development. Only upload **public** PDFs —
> free-tier content may be used to improve Google's models.
