from typing import List
from models.device import SmartHomeDevice
from db_backends.base_db_backend import ABaseDbBackend
from embeddings.base_llm_embedding import ABaseEmbedding
from llm_backends.base_backend import ALlmBaseBackend

class LlmAgent:
    def __init__(self, db_backend: ABaseDbBackend, embedding: ABaseEmbedding, llm_backend: ALlmBaseBackend) -> None:
        self.db_backend = db_backend
        self.embedding = embedding
        self.llm_backend = llm_backend
        
    def _get_devices(self, query: str) -> List[SmartHomeDevice]:
        embedded_query = self.embedding.embed_text(query)
        return self.db_backend.retrieve_devices(embedded_query)
    
    def _build_prompt(self, query: str) -> str:
        pass
        
    def process_user_query(self, query: str):
        devices = self._get_devices(query)