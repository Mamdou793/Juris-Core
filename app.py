import streamlit as st
import os
import time
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

# 2. Initialize Agents (Cached so they don't reload every click)
@st.cache_resource
def init_services():
    try:
        # 1. Ambient Identity (The 'FaceID' of the App)
        vault_url = os.getenv("AZURE_KEYVAULT_URL")
        credential = DefaultAzureCredential() 
        
        vault_client = SecretClient(vault_url=vault_url, credential=credential)
        
        openai_key = vault_client.get_secret("OPENAI-KEY").value
        search_key = vault_client.get_secret("SEARCH-KEY").value 
        doc_intel_key = vault_client.get_secret("DOC-INTEL-KEY").value

    except Exception as e:
        st.error("Identity Authentication Failed. Please run 'az login' in your terminal.")
        st.stop()
    
    # 3. Initialize with Vaulted keys
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
            HistoryService() # Already handles its own vault connection internally
        )

researcher, critic, gpt_client, ingestor, history = init_services()

# 3. Sidebar for Project Info
with st.sidebar:
    st.title("Juris Core")
    st.info("Legal Assistant & Document Analyzer")
    
    st.divider()
    
    # NEW: File Uploader Section
    st.subheader("📁 Data Ingestion")
    uploaded_file = st.file_uploader("Upload new document", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file:
        if st.button("Process & Index"):
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                # Read the file bytes
                file_bytes = uploaded_file.read()
                num_chunks = ingestor.process_and_upload(file_bytes, uploaded_file.name)
                st.success(f"Indexed {num_chunks} chunks!")
                time.sleep(2) # Brief pause for user to see success
                st.rerun()

    st.divider()
    st.write("**Current Index:**", os.getenv("AZURE_SEARCH_INDEX_NAME"))
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 4. Chat Interface
st.title("Juris Core Analyst")

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_default_session"
    
if "messages" not in st.session_state:
    # Fetch from cloud
    cloud_history = history.get_history(st.session_state.session_id)
    
    if cloud_history:
        # Update messages state
        st.session_state.messages = [{"role": item["role"], "content": item["content"]} for item in cloud_history]
        
        # --- AUDIT CHECK ---
        # We check the 'cloud_history' list we just pulled
        is_authentic, auth_message = history.verify_history_integrity(cloud_history)
        
        # We store the result in session_state so it persists on every refresh
        st.session_state.auth_status = (is_authentic, auth_message)
    else:
        st.session_state.messages = []
        st.session_state.auth_status = (True, "No records yet.")

# --- DISPLAY AUDIT STATUS ---
# Place this OUTSIDE the 'if' block so it stays in the sidebar at all times
if "auth_status" in st.session_state:
    is_authentic, auth_message = st.session_state.auth_status
    with st.sidebar:
        st.divider()
        if not is_authentic:
            st.error(f"🛑 {auth_message}")
        else:
            st.success("🛡️ Records Verified: Authentic")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about your legal documents..."):
    # Add user message to history
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
            system_message = (
                "You are the Juris-Omni Legal Analyst. Provide high-fidelity answers "
                "based ONLY on the provided context. Cite sources in brackets [File.md]."
            )
            response = gpt_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Context documents:\n{context_text}\n\nQuestion: {prompt}"}
                ]
            )
            draft = response.choices[0].message.content
            st.write("Draft complete.")

        with st.status("⚖️ Critic verifying...", expanded=False):
            critic_feedback = critic.review_response(prompt, context_text, draft)
            st.write("Review finished.")

        # Final Logic Display
        if "PASSED" in critic_feedback.upper():
            final_content = draft
            st.markdown(final_content)
        else:
            final_content = draft # Or combine with feedback if you prefer
            st.warning("The Critic found issues with this analysis.")
            with st.expander("See Critic Feedback"):
                st.write(critic_feedback)
            st.markdown(final_content)
        
        # SAVE ASSISTANT RESPONSE TO CLOUD
        st.session_state.messages.append({"role": "assistant", "content": final_content})
        history.save_message(st.session_state.session_id, "assistant", final_content)