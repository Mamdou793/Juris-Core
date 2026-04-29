import streamlit as st
import os
import time
import pandas as pd
from dotenv import load_dotenv
from agents.researcher.researcher import ResearcherAgent
from agents.critic.critic import CriticAgent
from openai import AzureOpenAI
from services.ingestion_service import IngestionService
from services.history_service import HistoryService
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# 1. Page Configuration
st.set_page_config(page_title="Juris Core", page_icon="⚖️", layout="wide")
load_dotenv()

# 2. Initialize Agents
@st.cache_resource
def init_services():
    try:
        vault_url = os.getenv("AZURE_KEYVAULT_URL")
        credential = DefaultAzureCredential() 
        vault_client = SecretClient(vault_url=vault_url, credential=credential)
        
        openai_key = vault_client.get_secret("OPENAI-KEY").value
        search_key = vault_client.get_secret("SEARCH-KEY").value 
        doc_intel_key = vault_client.get_secret("DOC-INTEL-KEY").value

    except Exception as e:
        st.error("Identity Authentication Failed. Please run 'az login' in your terminal.")
        st.stop()
    
    return (
            ResearcherAgent(search_key=search_key, openai_key=openai_key), 
            CriticAgent(openai_key=openai_key), 
            AzureOpenAI(
                api_key=openai_key,
                api_version="2024-06-01",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            ), 
            IngestionService(
                doc_intel_key=doc_intel_key, 
                search_key=search_key, 
                openai_key=openai_key
            ), 
            HistoryService()
        )

researcher, critic, gpt_client, ingestor, history = init_services()

# 3. Sidebar for Project Info
with st.sidebar:
    st.title("Juris Core")
    st.info("Legal Assistant & Document Analyzer")
    st.divider()
    
    st.subheader("📁 Data Ingestion")
    uploaded_file = st.file_uploader("Upload new document", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file:
        if st.button("Process & Index"):
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                file_bytes = uploaded_file.read()
                num_chunks = ingestor.process_and_upload(file_bytes, uploaded_file.name)
                st.success(f"Indexed {num_chunks} chunks!")
                time.sleep(2)
                st.rerun()

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 4. State Management
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_default_session"
    
if "messages" not in st.session_state:
    cloud_history = history.get_history(st.session_state.session_id)
    if cloud_history:
        st.session_state.messages = [{"role": item["role"], "content": item["content"]} for item in cloud_history]
        is_authentic, auth_message = history.verify_history_integrity(cloud_history)
        st.session_state.auth_status = (is_authentic, auth_message)
    else:
        st.session_state.messages = []
        st.session_state.auth_status = (True, "No records yet.")

# Sidebar Status Display
if "auth_status" in st.session_state:
    is_authentic, auth_message = st.session_state.auth_status
    with st.sidebar:
        st.divider()
        if not is_authentic:
            st.error(f"🛑 {auth_message}")
        else:
            st.success("🛡️ Records Verified: Authentic")

# --- 5. TABS INTERFACE ---
tab_chat, tab_audit = st.tabs(["💬 Juris Analyst", "🛡️ Security Dashboard"])

with tab_chat:
    st.title("Juris Core Analyst")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask about your legal documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        history.save_message(st.session_state.session_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("🕵️ Researcher hunting for facts...", expanded=False):
                search_results = researcher.search_knowledge_base(prompt)
                context_text = ""
                for i, res in enumerate(search_results):
                    context_text += f"\n-- DOCUMENT {i+1}: {res['source']} --\n{res['content']}\n"
                st.write("Found relevant clauses.")

            with st.status("🧠 Analyst synthesizing...", expanded=False):
                system_message = "You are the Juris-Omni Legal Analyst. Provide answers based ONLY on context."
                response = gpt_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"}
                    ]
                )
                draft = response.choices[0].message.content
                st.write("Draft complete.")

            with st.status("⚖️ Critic verifying...", expanded=False):
                critic_feedback = critic.review_response(prompt, context_text, draft)
                st.write("Review finished.")

            if "PASSED" in critic_feedback.upper():
                final_content = draft
            else:
                final_content = draft
                st.warning("The Critic found issues.")
                with st.expander("See Critic Feedback"):
                    st.write(critic_feedback)
            
            st.markdown(final_content)
            st.session_state.messages.append({"role": "assistant", "content": final_content})
            history.save_message(st.session_state.session_id, "assistant", final_content)

with tab_audit:
    st.title("Security & Integrity Audit")
    
    # 1. System Health Metrics
    st.subheader(" 🔑Identity & Encryption Status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Identity Mode", "Managed ID", "Active")
    c2.metric("Vault Key", "Key Vault", "Connected")
    c3.metric("DB Encryption", "AES-128", "E2EE")

    st.divider()

    # 2. Cryptographic Notary Logs
    st.subheader("📜 Cryptographic Notary Logs")
    st.info("Re-verifying all database records against the Vaulted SHA-256 salt...")
    
    audit_data = history.get_audit_logs(st.session_state.session_id)
    
    if audit_data:
        df = pd.DataFrame(audit_data)
        st.dataframe(df, use_container_width=True)
        
        tampered_count = sum(1 for d in audit_data if "🛑" in d["Status"])
        if tampered_count == 0:
            st.success(f"Integrity Audit: {len(audit_data)} records verified authentic.")
        else:
            st.error(f"ALERT: {tampered_count} records failed integrity check!")
    else:
        st.write("No database records available for auditing.")

    st.divider()
    
    # 3. Encryption Proof (Visualizing "The Blinding")
    st.subheader("The Blinding Proof")
    st.write("Comparing Ciphertext (stored in Azure) vs. Plaintext (on this device).")
    if audit_data:
        sample = audit_data[0]
        col_cloud, col_local = st.columns(2)
        with col_cloud:
            st.code(sample['Encrypted_Preview'], language="text")
            st.caption("Ciphertext (What Azure sees)")
        with col_local:
            st.code(sample['Decrypted_Preview'], language="text")
            st.caption("Plaintext (What You see)")