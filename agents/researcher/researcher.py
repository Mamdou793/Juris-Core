import os
from typing import List, Dict
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType, QueryCaptionType, QueryAnswerType
from openai import AzureOpenAI

class ResearcherAgent:
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

    def search_knowledge_base(self, query: str, top_k: int = 15) -> List[Dict]:
        """
        Executes a Semantic Hybrid Search. 
        Combines Vector + Keyword + Semantic Reranking for high-density documents.
        """
        query_vector = self._get_embedding(query)
        
        # Define the Vector component
        vector_query = VectorizedQuery(
            vector=query_vector, 
            k_nearest_neighbors=top_k, 
            fields="content_vector"
        )

        try:
            # Perform Hybrid + Semantic Search
            results = self.search_client.search(
                search_text=query, # Provides the Keyword component
                vector_queries=[vector_query], # Provides the Vector component
                query_type=QueryType.SEMANTIC, # Activates the Semantic Ranker
                semantic_configuration_name="default", # ENSURE THIS MATCHES YOUR PORTAL NAME
                query_answer=QueryAnswerType.EXTRACTIVE, # AI tries to find a direct answer string
                query_caption=QueryCaptionType.EXTRACTIVE, # Highlight relevant snippets
                top=top_k
            )
        except Exception as e:
            # Fallback to standard vector search if Semantic is not yet enabled in Portal
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query]
            )

        return [
            {
                "source": res['title'],
                "content": res['content'],
                "score": res['@search.score']
            } for res in results
        ]