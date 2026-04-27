import os
from azure.cosmos import CosmosClient, PartitionKey
from datetime import datetime

class HistoryService:
    def __init__(self):
        url = os.getenv("AZURE_COSMOS_ENDPOINT")
        key = os.getenv("AZURE_COSMOS_KEY")
        
        if not url or not key:
            # Fallback or alert if not configured
            self.client = None
            return

        self.client = CosmosClient(url, key)
        self.db = self.client.create_database_if_not_exists(id="JurisOmni")
        self.container = self.db.create_container_if_not_exists(
            id="ChatHistory", 
            partition_key=PartitionKey(path="/session_id")
        )

    def save_message(self, session_id, role, content):
        if not self.client: return
        
        item = {
            "id": f"{session_id}_{datetime.utcnow().timestamp()}",
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.container.create_item(body=item)

    def get_history(self, session_id):
        if not self.client: return []
        
        query = "SELECT * FROM c WHERE c.session_id = @session_id ORDER BY c.timestamp ASC"
        params = [{"name": "@session_id", "value": session_id}]
        
        items = list(self.container.query_items(
            query=query, 
            parameters=params, 
            enable_cross_partition_query=False
        ))
        return items