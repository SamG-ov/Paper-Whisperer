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
| LLM + Embeddings | Google Gemini (`gemini-2.0-flash`, `text-embedding-004`) |
| Vector Database  | ChromaDB                                           |
| UI               | Streamlit                                          |
| Packaging        | Docker                                             |

## 🚧 Status

Under active development — built milestone by milestone. See the roadmap below.

## 🗺️ Roadmap

- [ ] **M0** — Project foundations & setup
- [ ] **M1** — PDF ingestion (load & extract text)
- [ ] **M2** — Chunking
- [ ] **M3** — Embeddings + vector store
- [ ] **M4** — Retrieval
- [ ] **M5** — Generation with citations
- [ ] **M6** — Streamlit UI
- [ ] **M7** — Evaluation & observability
- [ ] **M8** — Advanced retrieval (re-ranking, hybrid search, caching)
- [ ] **M9** — Dockerize & deploy

## ⚙️ Setup

```powershell
# 1. Create and activate a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
py -m pip install -r requirements.txt

# 3. Configure secrets
copy .env.example .env
# then edit .env and add your Gemini API key
```

> ⚠️ This project uses Gemini's free tier for development. Only upload **public** PDFs —
> free-tier content may be used to improve Google's models.
