import streamlit as st
import os
import time
from dotenv import load_dotenv
from agents.researcher.researcher import ResearcherAgent
from agents.critic.critic import CriticAgent
from openai import AzureOpenAI
from services.ingestion_service import IngestionService
from services.history_service import HistoryService

# 1. Page Configuration
st.set_page_config(page_title="Juris Core", page_icon="⚖️", layout="wide")
load_dotenv()

# 2. Initialize Agents (Cached so they don't reload every click)
@st.cache_resource
def init_services():
    return ResearcherAgent(), CriticAgent(), AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-06-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    ), IngestionService(), HistoryService()

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
    cloud_history = history.get_history(st.session_state.session_id)
    if cloud_history:
        st.session_state.messages = [{"role": item["role"], "content": item["content"]} for item in cloud_history]
    else:
        st.session_state.messages = []

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