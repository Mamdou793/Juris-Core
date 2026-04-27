import streamlit as st
import os
from dotenv import load_dotenv
from agents.researcher.researcher import ResearcherAgent
from agents.critic.critic import CriticAgent
from openai import AzureOpenAI

# 1. Page Configuration
st.set_page_config(page_title="Juris-Omni Core", page_icon="⚖️", layout="wide")
load_dotenv()

# 2. Initialize Agents (Cached so they don't reload every click)
@st.cache_resource
def init_agents():
    return ResearcherAgent(), CriticAgent(), AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-06-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

researcher, critic, gpt_client = init_agents()

# 3. Sidebar for Project Info
with st.sidebar:
    st.title("⚖️ Juris-Omni")
    st.info("Legal Assistant & Document Analyzer")
    st.divider()
    st.write("**Current Index:**", os.getenv("AZURE_SEARCH_INDEX_NAME"))
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 4. Chat Interface
st.title("Juris-Omni-Core Analyst")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about your legal documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
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
            st.markdown(draft)
            st.session_state.messages.append({"role": "assistant", "content": draft})
        else:
            st.warning("The Critic found issues with this analysis.")
            with st.expander("See Critic Feedback"):
                st.write(critic_feedback)
            st.markdown(draft)
            st.session_state.messages.append({"role": "assistant", "content": draft})