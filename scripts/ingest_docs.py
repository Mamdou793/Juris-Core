import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

def analyze_to_markdown(file_path):
    print(f"🚀 Analyzing: {file_path}")
    
    client = DocumentIntelligenceClient(
        endpoint=os.getenv("DOC_INTEL_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("DOC_INTEL_KEY"))
    )

    with open(file_path, "rb") as f:
        # NOTICE: Changed 'analyze_request' to 'body'
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout", 
            body=AnalyzeDocumentRequest(bytes_source=f.read()), # Fixed here
            output_content_format="markdown" 
        )
    
    result = poller.result()
    return result.content

if __name__ == "__main__":
    # Ensure folders exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Automatically find the first PDF in data/raw
    raw_files = [f for f in os.listdir("data/raw") if f.endswith((".pdf", ".docx"))]
    
    if not raw_files:
        print("❌ No PDF or docx found in data/raw/. Please drop a file there!")
    else:
        # Process the first one found
        target_file = os.path.join("data/raw", raw_files[0])
        md_data = analyze_to_markdown(target_file)
        
        # Save using the original name but with .md extension
        base_name = os.path.splitext(raw_files[0])[0]
        output_path = f"data/processed/{base_name}.md"
        
        with open(output_path, "w") as f:
            f.write(md_data)
        
        print(f"✅ Success! {raw_files[0]} converted to {output_path}")