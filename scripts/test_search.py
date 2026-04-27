import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

load_dotenv()

# Initialize Clients
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

aoai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def get_embedding(text):
    return aoai_client.embeddings.create(input=[text], model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")).data[0].embedding

def semantic_query(query_text):
    print(f"🔍 Searching for: '{query_text}'...")
    
    # 1. Turn the user's question into a vector
    query_vector = get_embedding(query_text)
    
    # 2. Perform the Vector Search
    results = search_client.search(
        search_text=None, # Pure vector search
        vector_queries=[{
            "vector": query_vector,
            "fields": "content_vector",
            "k": 3, # Get top 3 chunks
            "kind": "vector"
        }]
    )

    for result in results:
        print(f"\n📄 Source: {result['title']}")
        print(f"📊 Score: {result['@search.score']}")
        print(f"📝 Content: {result['content'][:200]}...")

if __name__ == "__main__":
    # Change this query to something that is actually in your Project Status Report!
    user_query = "What is the current status of the project?" 
    semantic_query(user_query)