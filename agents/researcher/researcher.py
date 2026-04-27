import os
from typing import List, Dict
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

load_dotenv()

class ResearcherAgent:
    def __init__(self):
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
        self.aoai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    def _get_embedding(self, text: str) -> List[float]:
        return self.aoai_client.embeddings.create(
            input=[text], 
            model=self.embedding_model
        ).data[0].embedding

    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict]:
        """Returns structured results with source metadata."""
        query_vector = self._get_embedding(query)
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[{
                "vector": query_vector,
                "fields": "content_vector",
                "k": top_k,
                "kind": "vector"
            }]
        )

        # Return as a list of dictionaries for the Analyst to process
        return [
            {
                "source": res['title'],
                "content": res['content'],
                "score": res['@search.score']
            } for res in results
        ]
        # Format results into a single string for the next agent to read
        context_block = "\n--- SEARCH RESULTS ---\n"
        for res in results:
            context_block += f"Source: {res['title']}\nContent: {res['content']}\n\n"
        
        return context_block