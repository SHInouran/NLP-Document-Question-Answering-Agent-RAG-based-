from typing import Any, List, Dict
from Embedding_model import EmbeddingModel
from Vector_database import VectorDatabase


class Retriever :
    def __init__(self,db_path: str="./chroma_db"):
        self.db_manager = VectorDatabase(db_path=db_path)

    def retrieve(self, query : str, model_key: str = "minilm", top_k: int = 3) ->List[Dict[str, Any]]:
        embedder = EmbeddingModel(model_key =model_key)
        query_vect = embedder.embedd(query)
        raw_results = self.db_manager.query_similarity( model_key = model_key, query_embedding= query_vect,n_results=top_k)
        formatted_documents= []
        docs = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metadatas,distances):
            formatted_documents.append({
                "text": text,
                "metadata": meta,
                "score": round(1.0 -dist, 4) #converting distance to a score
            })

        return formatted_documents

if __name__ == "__main__":
    # Test retriever logic
    print("--- Testing Retriever Module ---")

    # Initialize retriever
    retriever = Retriever(db_path="./test_chroma_db")

    # Run a test query against our previous test database
    results = retriever.retrieve(
        query="What is a Pod in Kubernetes?", 
        model_key="minilm", 
        top_k=2
    )

    print(f"\nRetrieved {len(results)} context matches:")
    for idx, doc in enumerate(results, 1):
        print(f"\nResult #{idx} (Score: {doc['score']}):")
        print(f"Source: {doc['metadata'].get('source_file')}")
        print(f"Title:  {doc['metadata'].get('title')}")
        print(f"Text:   {doc['text']}")