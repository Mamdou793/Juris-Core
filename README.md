# Juris-Omni Core: Legal Analyst Agentic Platform

An enterprise-grade Legal RAG (Retrieval-Augmented Generation) platform built with a multi-agent architecture to ensure high-fidelity document analysis and verification.

## Key Features
* **Multi-Agent Workflow:** Utilizes a **Researcher-Analyst-Critic** loop. The Critic agent cross-verifies all claims against retrieved context to eliminate hallucinations.
* **In-App Ingestion (The Scaler):** Streamlined sidebar uploader that parses PDFs/Docs into Markdown using **Azure Document Intelligence** and indexes them into **Azure AI Search**.
* **Persistent Cloud Memory:** Full chat history persistence using **Azure Cosmos DB (NoSQL)**, allowing session recovery across browser refreshes.
* **High-Fidelity RAG:** Hybrid search capabilities using Azure OpenAI embeddings.

## Technical Architecture
* **Frontend:** Streamlit
* **Orchestration:** Python (Custom Agent Logic)
* **LLMs:** Azure OpenAI (GPT-4o)
* **Vector Database:** Azure AI Search
* **Document Parsing:** Azure AI Document Intelligence (v4.0 Preview)
* **Session State:** Azure Cosmos DB

## Setup
1. Clone the repository.
2. Create a `.env` file based on the provided environment variables (Azure OpenAI, Search, Cosmos, and Doc Intel).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Run The Application
streamlit run app.py