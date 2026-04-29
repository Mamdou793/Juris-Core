# Juris Core: Secure Legal Analyst Agentic Platform With Security Dashboard

An enterprise-grade Legal RAG (Retrieval-Augmented Generation) platform built with a multi-agent architecture to ensure high-fidelity document analysis and verification with Security & Integrity Dashboard.

## Key Features
* **Multi-Agent Workflow:** Utilizes a **Researcher-Analyst-Critic** loop. The Critic agent cross-verifies all claims against retrieved context to eliminate hallucinations.
* **In-App Ingestion (The Scaler):** Streamlined sidebar uploader that parses PDFs/Docs into Markdown using **Azure Document Intelligence** and indexes them into **Azure AI Search**.
* **Persistent Cloud Memory:** Full chat history persistence using **Azure Cosmos DB (NoSQL)**, allowing session recovery across browser refreshes.
* **High-Fidelity RAG:** Hybrid search capabilities using Azure OpenAI embeddings.
* **Zero-Trust Identity**: Passwordless authentication using Azure Managed Identities (DefaultAzureCredential). No secrets are stored locally or in plain-text .env files.
* **End-to-End Encryption (The Blinding)**: All chat history is encrypted on the client side using AES-128 (Fernet) before reaching the cloud. Data is "blind" to the cloud provider.
* **Cryptographic Notary (SHA-256)**: Every record is signed with a digital fingerprint. The UI includes a real-time Integrity Guard that detects and flags unauthorized database tampering or "poisoning."

## Technical Architecture
* **Frontend:** Streamlit
* **Orchestration:** Python (Custom Agent Logic)
* **LLMs:** Azure OpenAI (GPT-4o-mini)
* **Vector Database:** Azure AI Search
* **Document Parsing:** Azure AI Document Intelligence (v4.0 Preview)
* **Session State:** Azure Cosmos DB
* **Secret Management:** All API keys (OpenAI, Search, Cosmos) are centralized in Azure Key Vault.
* **Data Sovereignty:** By using Client-Side Encryption, the system ensures that even a breach of the database layer yields only unreadable ciphertext.
* **Audit Logging:** Every message retrieval triggers a verification handshake, comparing stored hashes against a vaulted secret salt to ensure 100% evidentiary integrity.

## Setup
1. Clone the repository.
2. Identity: Ensure you are authenticated via Azure CLI: az login.
3. Create a .env file with Public Endpoints only (No Keys):
4. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Run The Application
streamlit run app.py
