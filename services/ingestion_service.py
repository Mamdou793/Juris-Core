import os
import time
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

class IngestionService:
    def __init__(self):
        # Ensure we use consistent env names
        key = os.getenv("DOC_INTEL_KEY")
        endpoint = os.getenv("DOC_INTEL_ENDPOINT")
        
        if not key or not endpoint:
            raise ValueError("Missing DOC_INTEL credentials in .env")

        self.doc_client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
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

    def process_and_upload(self, file_content, filename):
        # 1. Analyze with Document Intelligence
        # Using output_content_format="markdown" specifically for this SDK version
        poller = self.doc_client.begin_analyze_document(
            model_id="prebuilt-layout",
            analyze_request=file_content,
            output_content_format="markdown", # This is the key fix
            content_type="application/octet-stream"
        )
        result = poller.result()
        md_content = result.content

        # 2. Chunking (Simple split for now)
        chunks = [md_content[i:i+4000] for i in range(0, len(md_content), 4000)]
        
        batch = []
        for i, chunk in enumerate(chunks):
            embedding = self.aoai_client.embeddings.create(
                input=[chunk], 
                model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            ).data[0].embedding
            
            # Sanitize ID to ensure it only contains valid characters for Azure Search
            clean_id = f"id_{int(time.time())}_{i}" 
            
            batch.append({
                "id": clean_id,
                "title": filename,
                "content": chunk,
                "content_vector": embedding
            })

        # 3. Upload to Azure
        self.search_client.upload_documents(documents=batch)
        return len(batch)