import os
import time
import uuid
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SimpleField,
    SearchableField
)
from azure.storage.blob import BlobServiceClient
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

        # 2. Search Client Setup (Update to SearchIndexClient for creation)
        s_key = search_key or os.getenv("AZURE_SEARCH_KEY")
        self.search_key = s_key
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(s_key)
        )

        # 3. OpenAI Client Setup
        o_key = openai_key or os.getenv("AZURE_OPENAI_KEY")
        self.aoai_client = AzureOpenAI(
            api_key=o_key,
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        
        # 4. Azure Blob Storage Setup
        connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.blob_service_client = None
        self.container_name = "juris-documents"
        
        if connect_str:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(connect_str)
                container_client = self.blob_service_client.get_container_client(self.container_name)
                if not container_client.exists():
                    container_client.create_container()
            except Exception:
                pass

    def get_or_create_index(self):
        """Creates the search index with semantic configuration if it doesn't exist."""
        from azure.search.documents.indexes import SearchIndexClient
        
        # Using SearchIndexClient for administrative tasks (like index creation)
        index_client = SearchIndexClient(
            endpoint=self.search_endpoint, 
            credential=AzureKeyCredential(self.search_key)
        )
        
        # Check if index exists
        indices = index_client.list_index_names()
        if self.index_name in indices:
            print(f"Index {self.index_name} already exists.")
            return

        # Define Schema
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.microsoft", facetable=True, filterable=True, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                        searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
        ]

        # Vector Config
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
            profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
        )

        # Semantic Config
        semantic_config = SemanticConfiguration(
            name="mySemanticConfig",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
                title_field=SemanticField(field_name="title")
            )
        )

        # Create Index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_configurations=[semantic_config]
        )
        
        index_client.create_index(index)
        print(f"Index {self.index_name} created successfully.")

    def process_and_upload(self, file_content, filename):
        """Robust ingestion with batching and proactive index creation for large documents."""
        
        # 1. Ensure the index exists before processing
        self.get_or_create_index()

        # 2. Save original file to Azure Blob Storage if connected
        if self.blob_service_client:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=filename
            )
            blob_client.upload_blob(file_content, overwrite=True)

        # 3. Analyze with Document Intelligence (Still blocking, unavoidable)
        print(f"Starting Document Intelligence on {filename} (700 pages takes time)...")
        poller = self.doc_client.begin_analyze_document(
            model_id="prebuilt-layout",
            analyze_request=file_content,
            output_content_format="markdown",
            content_type="application/octet-stream"
        )
        result = poller.result()
        md_content = result.content
        print(f"Extraction complete ({len(md_content)} characters).")

        # 4. Chunking Logic
        chunk_size = 4000
        chunks = [md_content[i:i+chunk_size] for i in range(0, len(md_content), chunk_size)]
        total_chunks = len(chunks)
        print(f"Document split into {total_chunks} chunks.")
        
        # 5. Process and upload chunks in small batches
        batch_size = 25 # Process 25 chunks at a time to prevent timeouts
        uploaded_count = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            documents_batch = []
            
            # Start of batch processing
            start_time = time.time()
            print(f"Processing batch {i//batch_size + 1}: Chunks {i+1} to {min(i+batch_size, total_chunks)}...")
            
            # Process each chunk in the batch sequentially (can't batch embeddings easily)
            for j, chunk in enumerate(batch):
                try:
                    embedding = self.aoai_client.embeddings.create(
                        input=[chunk], 
                        model=self.embedding_model
                    ).data[0].embedding
                    
                    clean_id = f"id_{int(time.time())}_{uuid.uuid4().hex[:6]}" 
                    
                    documents_batch.append({
                        "id": clean_id,
                        "title": filename,
                        "content": chunk,
                        "content_vector": embedding
                    })
                except Exception as e:
                    print(f"Error creating embedding for chunk {i+j+1}: {e}")
                    continue # Skip this chunk but keep processing

            # Immediately upload the small batch of completed chunks
            if documents_batch:
                try:
                    self.search_client.upload_documents(documents=documents_batch)
                    uploaded_count += len(documents_batch)
                    elapsed = time.time() - start_time
                    print(f"Successfully uploaded batch. Total chunks in index: {uploaded_count}. (Time: {elapsed:.2f}s)")
                except Exception as e:
                    print(f"Error uploading batch: {e}")

        return uploaded_count