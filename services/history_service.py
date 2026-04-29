import os
from datetime import datetime
import hashlib
from cryptography.fernet import Fernet
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

class HistoryService:
    def __init__(self):
        vault_url = os.getenv("AZURE_KEYVAULT_URL")
        credential = DefaultAzureCredential()
        
        try:
            self.vault_client = SecretClient(vault_url=vault_url, credential=credential)
            url = os.getenv("AZURE_COSMOS_ENDPOINT")
            key = self.vault_client.get_secret("COSMOS-KEY").value
            self.secret_salt = self.vault_client.get_secret("AUDIT-SECRET-KEY").value
            
            # --- NEW: INITIALIZE THE CRYPTOR ---
            # We pull the encryption key from the vault
            raw_encryption_key = self.vault_client.get_secret("MASTER-ENCRYPTION-KEY").value
            self.cipher = Fernet(raw_encryption_key.encode())
            
        except Exception as e:
            print(f"Vault Connection Failed: {e}")
            self.client = None
            return

        self.client = CosmosClient(url, key)
        self.db = self.client.create_database_if_not_exists(id="JurisOmni")
        self.container = self.db.create_container_if_not_exists(
            id="ChatHistory", 
            partition_key=PartitionKey(path="/session_id")
        )

    def _encrypt(self, text: str) -> str:
        """Turns plain text into unreadable ciphertext."""
        return self.cipher.encrypt(text.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        """Turns unreadable ciphertext back into plain text."""
        return self.cipher.decrypt(ciphertext.encode()).decode()

    def _create_fingerprint(self, content, timestamp, role):
        # IMPORTANT: We hash the PLAIN text content to keep our audit trail logic clean
        raw_string = f"{content}|{timestamp}|{role}|{self.secret_salt}"
        return hashlib.sha256(raw_string.encode()).hexdigest()

    def save_message(self, session_id, role, content):
        if not self.client: return
        
        timestamp = datetime.utcnow().isoformat()
        audit_hash = self._create_fingerprint(content, timestamp, role)
        
        # --- ENCRYPT THE CONTENT ---
        encrypted_content = self._encrypt(content)
        
        item = {
            "id": f"{session_id}_{datetime.utcnow().timestamp()}",
            "session_id": session_id,
            "role": role,
            "content": encrypted_content, # The cloud only sees this!
            "timestamp": timestamp,
            "audit_hash": audit_hash 
        }
        
        self.container.create_item(body=item)
        
    def get_history(self, session_id):
        if not self.client: return []
        
        query = "SELECT * FROM c WHERE c.session_id = @session_id ORDER BY c.timestamp ASC"
        params = [{"name": "@session_id", "value": session_id}]
        
        items = list(self.container.query_items(query=query, parameters=params, enable_cross_partition_query=False))
        
        # --- DECRYPT THE CONTENT FOR THE UI ---
        for item in items:
            item['content'] = self._decrypt(item['content'])
            
        return items

    def verify_history_integrity(self, items):
        # This remains the same because we are verifying against the decrypted content
        for item in items:
            stored_hash = item.get("audit_hash")
            calculated_hash = self._create_fingerprint(item['content'], item['timestamp'], item['role'])
            if stored_hash != calculated_hash:
                return False, f"Integrity Breach detected at {item['timestamp']}!"
        return True, "All records verified authentic."