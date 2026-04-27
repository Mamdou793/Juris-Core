# Juris-Core

An Agentic RAG (Retrieval-Augmented Generation) Legal Assistant built with **Azure AI Search**, **OpenAI GPT-4o-mini**, and **Streamlit**.

## Architecture
Juris-Omni-Core uses a multi-agent orchestration pattern to ensure high-fidelity legal analysis:
- **Researcher Agent:** Performs vector search across indexed legal documents.
- **Analyst Agent:** Synthesizes findings with mandatory inline citations.
- **Critic Agent:** A deterministic "Zero-Trust" layer that validates the Analyst's output against the raw source text to prevent hallucinations.

## Tech Stack
- **Orchestration:** Python (Sequential Agentic Chain)
- **LLM:** Azure OpenAI (GPT-4o-mini & Text-Embedding-3-Small)
- **Vector Store:** Azure AI Search (HNSW Algorithm)
- **UI:** Streamlit
- **Document Processing:** Azure AI Document Intelligence

## Getting Started
1. **Environment:** Setup your `.env` with Azure credentials (see `.env.example`).
2. **Ingest:** Place documents in `data/raw` and run `python scripts/ingest_docs.py`.
3. **Index:** Run `python scripts/vectorize_docs.py` to populate the vector store.
4. **Launch:** Run `streamlit run app.py`.
