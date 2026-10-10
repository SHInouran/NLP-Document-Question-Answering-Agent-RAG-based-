from sentence_transformers import SentenceTransformer
from typing import List,Dict

class EmbeddingModel:
    MODELS: Dict[str,Dict[str,any]]={
        "minilm":{
            "name":"sentence-transformers/all-MiniLM-L6-v2",
            "dim" : 384,
            "description":"A small, fast, and efficient model for generating sentence embeddings."
        },
        "BGE-M3":{
            "name": "BAAI/bge-m3",
            "dim": 1024,
            "description": "A large-scale model for generating high-quality sentence embeddings."
        },
        "JINA":{
            "name" :"jinaai/jina-embeddings-v2-base-en",
            "dim": 768,
            "description": "A model optimized for generating embeddings for various NLP tasks." 
        }
    }
    def __init__(self, model_key: str = "minilm" , device: str= "cuda"):
        if model_key not in self.MODELS:
            raise ValueError (f"Model'{model_key}'not supported")
        self.model_key = model_key
        self.device = device 
        self.model_info = self.MODELS[model_key]

        trust_remote_code = True if model_key =="JINA" else False
        self.model = SentenceTransformer(self.model_info["name"],device = self.device , trust_remote_code = trust_remote_code)

    def embedd (self, query: str ) -> List[float]:
        embedding = self.model.encode( query, normalize_embeddings=True, convert_to_numpy=True)
        return embedding.tolist()
    def embedd_batch (self, text: List[str], batch_size: int=32) -> List[List[float]]:
        embeddings = self.model.encode(text, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.tolist()
    def get_dimensions(self) -> int:
        """Returns the expected vector length for the active model."""
        return self.model_info["dim"]

if __name__ == "__main__":
    print('testing embedding model')
    text = " nan should text kiwi "
    model = EmbeddingModel(model_key="minilm", device="cpu")
    embedding = model.embedd(text)
    print(f"Embedding for '{text}': {embedding}")
    