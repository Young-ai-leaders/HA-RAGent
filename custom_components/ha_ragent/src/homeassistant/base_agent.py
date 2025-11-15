from typing import List
from models.device import Device
from db_backends.base_db_backend import ABaseDbBackend
from embeddings.base_embedder import ABaseEmbedder
from llm_backends.base_backend import ALlmBaseBackend

class LlmAgent:
    def __init__(self, db_backend: ABaseDbBackend, embedding: ABaseEmbedder, llm_backend: ALlmBaseBackend) -> None:
        self.db_backend = db_backend
        self.embedding = embedding
        self.llm_backend = llm_backend
        
    def _get_devices(self, query: str) -> List[Device]:
        embedded_query = self.embedding.embed_text(query)
        return self.db_backend.retrieve_devices(embedded_query)
    
    def _build_prompt(self, query: str) -> str:
        pass
        
    def process_user_query(self, query: str):
        devices = self._get_devices(query)