import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

class CriticAgent:
    def __init__(self):
        self.gpt_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    def review_response(self, question: str, context: str, draft: str) -> str:
        """Inspects the draft for hallucinations or missing citations."""
        
        system_message = (
            "You are the Juris-Omni Legal Critic. Your job is to verify the Analyst's draft.\n\n"
            "CHECKLIST:\n"
            "1. Fact-Check: Is every claim in the draft supported by the provided Context?\n"
            "2. Citation Check: Does every fact have a bracketed source [File.md]?\n"
            "3. Hallucination Alert: Did the Analyst add info NOT found in the context?\n\n"
            "OUTPUT FORMAT:\n"
            "If the draft is perfect, reply with 'PASSED'.\n"
            "If there are issues, list them clearly as 'CONCERNS' and provide a 'SUGGESTED FIX'."
        )

        user_content = (
            f"ORIGINAL CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"ANALYST DRAFT:\n{draft}"
        )

        response = self.gpt_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ],
            temperature=0 # We want the Critic to be robotic and precise
        )
        
        return response.choices[0].message.content