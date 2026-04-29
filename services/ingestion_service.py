import os
import time
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

class IngestionService:
    def __init__(self, doc_intel_key: str = None, search_key: str = None, openai_key: str = None):
        # 1. Document Intelligence Setup
        d_key = doc_intel_key or os.getenv("DOC_INTEL_KEY")
        d_endpoint = os.getenv("DOC_INTEL_ENDPOINT")
        
        self.doc_client = DocumentIntelligenceClient(
            endpoint=d_endpoint,
            credential=AzureKeyCredential(d_key)
        )

        # 2. Search Client Setup
        s_key = search_key or os.getenv("AZURE_SEARCH_KEY")
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            credential=AzureKeyCredential(s_key)
        )

        # 3. OpenAI Client Setup
        o_key = openai_key or os.getenv("AZURE_OPENAI_KEY")
        self.aoai_client = AzureOpenAI(
            api_key=o_key,
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Deployment names remain as env vars because they aren't 'secrets' (they are config)
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    def process_and_upload(self, file_content, filename):
        # 1. Analyze with Document Intelligence (Now using the vaulted credentials)
        poller = self.doc_client.begin_analyze_document(
            model_id="prebuilt-layout",
            analyze_request=file_content,
            output_content_format="markdown",
            content_type="application/octet-stream"
        )
        result = poller.result()
        md_content = result.content

        # 2. Chunking Logic
        chunks = [md_content[i:i+4000] for i in range(0, len(md_content), 4000)]
        
        batch = []
        for i, chunk in enumerate(chunks):
            embedding = self.aoai_client.embeddings.create(
                input=[chunk], 
                model=self.embedding_model
            ).data[0].embedding
            
            clean_id = f"id_{int(time.time())}_{i}" 
            
            batch.append({
                "id": clean_id,
                "title": filename,
                "content": chunk,
                "content_vector": embedding
            })

        # 3. Upload to Azure Search
        self.search_client.upload_documents(documents=batch)
        return len(batch)