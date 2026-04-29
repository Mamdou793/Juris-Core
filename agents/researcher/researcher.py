import os
from typing import List, Dict
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

class ResearcherAgent:
    # We now pass the keys in. If they aren't passed, it falls back to env 
    # for local testing, but the Vaulted keys will take priority.
    def __init__(self, search_key: str = None, openai_key: str = None):
        
        # 1. Initialize Search Client
        s_key = search_key or os.getenv("AZURE_SEARCH_KEY")
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            credential=AzureKeyCredential(s_key)
        )
        
        # 2. Initialize OpenAI Client
        o_key = openai_key or os.getenv("AZURE_OPENAI_KEY")
        self.aoai_client = AzureOpenAI(
            api_key=o_key,
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

        return [
            {
                "source": res['title'],
                "content": res['content'],
                "score": res['@search.score']
            } for res in results
        ]