import os
import re
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from openai import AzureOpenAI

load_dotenv()

# --- CONFIGURATION ---
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Initialize Clients
search_index_client = SearchIndexClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

aoai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def create_index_if_not_exists():
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        # ADDED THIS: Ensure the index knows what 'title' is
        SearchableField(name="title", type=SearchFieldDataType.String, filterable=True), 
        SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )

    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    
    try:
        search_index_client.create_index(index)
        print(f"✅ Index '{INDEX_NAME}' created from scratch.")
    except Exception as e:
        # Check if the error is just that it already exists
        if "AlreadyInUse" in str(e) or "already exists" in str(e).lower():
            print(f"ℹ️ Index '{INDEX_NAME}' already exists. Proceeding to upload...")
        else:
            print(f"❌ Actual error creating index: {e}")

def get_embedding(text):
    text = text.replace("\n", " ")
    return aoai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding

def chunk_markdown(md_text):
    # Splits by any H1 or H2 header (# or ##)
    sections = re.split(r'\n(?=# )|\n(?=## )', md_text)
    return [s.strip() for s in sections if s.strip()]

def upload_documents():
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    )
    
    processed_dir = "data/processed"
    batch = []
    
    for filename in os.listdir(processed_dir):
        if filename.endswith(".md"):
            print(f"📤 Vectorizing {filename}...")
            with open(os.path.join(processed_dir, filename), "r") as f:
                content = f.read()
                chunks = chunk_markdown(content)
                
                for i, chunk in enumerate(chunks):
                    doc_id = f"{filename.replace('.', '_').replace(' ', '_')}_{i}"
                    vector = get_embedding(chunk)
                    batch.append({
                        "id": doc_id,
                        "title": filename,
                        "content": chunk,
                        "content_vector": vector
                    })
    
    if batch:
        search_client.upload_documents(documents=batch)
        print(f"🚀 Successfully uploaded {len(batch)} chunks to {INDEX_NAME}.")

if __name__ == "__main__":
    create_index_if_not_exists()
    upload_documents()