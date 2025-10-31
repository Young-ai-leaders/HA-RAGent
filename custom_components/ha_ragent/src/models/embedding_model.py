from dataclasses import dataclass

@dataclass
class EmbeddingModel:
    model_name: str
    embedding_size: int