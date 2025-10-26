class BaseEmbedding:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap