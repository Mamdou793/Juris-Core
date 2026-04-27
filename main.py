import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from agents.researcher.researcher import ResearcherAgent
from agents.critic.critic import CriticAgent

load_dotenv()

# Initialize our components
researcher = ResearcherAgent()
critic = CriticAgent()
gpt_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def run_juris_omni(user_question: str):
    # --- PHASE 1: RESEARCH ---
    print("🕵️ Researcher is hunting for facts...")
    search_results = researcher.search_knowledge_base(user_question)
    
    context_text = ""
    for i, res in enumerate(search_results):
        context_text += f"\n-- DOCUMENT {i+1}: {res['source']} --\n{res['content']}\n"

    # --- PHASE 2: ANALYSIS (Drafting) ---
    print("🧠 Analyst is synthesizing across documents...")
    system_message = (
        "You are the Juris-Omni Legal Analyst. Your goal is to provide high-fidelity answers "
        "based ONLY on the provided context. \n\n"
        "STRICT RULES:\n"
        "1. Always cite the document name in brackets, e.g., [File_Name.md], next to the facts you mention.\n"
        "2. If documents conflict, highlight the discrepancy.\n"
        "3. If the answer is not in the context, state that clearly."
    )

    response = gpt_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context documents:\n{context_text}\n\nQuestion: {user_question}"}
        ]
    )
    draft_response = response.choices[0].message.content

    # --- PHASE 3: CRITICISM (Verification) ---
    print("⚖️ Critic is verifying the analysis...")
    # Passing the original question, the raw context, and the analyst's draft
    critic_feedback = critic.review_response(user_question, context_text, draft_response)

    if "PASSED" in critic_feedback.upper():
        print("\n✅ Verification Successful.")
        print("\n--- FINAL ANALYSIS ---")
        print(draft_response)
    else:
        print("\n⚠️ Quality Warning from Critic:")
        print(critic_feedback)
        # We still print the draft so you can see what failed
        print("\n--- UNVERIFIED DRAFT ---")
        print(draft_response)

if __name__ == "__main__":
    print("⚖️ Juris-Omni-Core is online.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("👤 Question: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        run_juris_omni(user_input)
        print("\n" + "="*50 + "\n")