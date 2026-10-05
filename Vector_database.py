import chromadb
from typing import Any, List,Dict

class VectorDatabase:
    def __init__(self, db_path:str = "./chroma_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=self.db_path)
        print(f"connected to chroma db at : {self.db_path}")

    def get_create_collection(self,model_key: str):
        collection_name = f"k8s_docs_{model_key}"
        collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        return collection

    def add_documents(self, model_key: str, texts:List[str] ,embeddings: List[List[float]], metadatas: List[Dict[str,Any]] , ids: List[str]) :
        collection = self.get_create_collection(model_key) 
        print(f"adding{len(texts)} items to collection '{collection.name}")
        collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        print(f"success adding documents to collection '{collection.name}'")

    def query_similarity( self, model_key: str, query_embedding: List[float], n_results: int= 5) -> List[Dict[str,Any]]:
        collection = self.get_create_collection(model_key)
        if collection.count() == 0 :
            print(f"warning : collection '{collection.name}' is empty")
            return {"documents":[[]], "metadatas": [[]] , "distances": [[]]}

        results = collection.query(query_embedding = [query_embedding], n_results = n_results)
        return results

if __name__ == "__main__":
   # Quick test to verify vector storage and search
    print("--- Testing Vector Database Manager ---")
    
    db_manager = VectorDatabase(db_path="./test_chroma_db")
    
    # Dummy data using 384-dim mock vectors
    test_model_key = "minilm"
    sample_texts = ["Pods run containers in Kubernetes.", "Services expose pods to network traffic."]
    sample_embeddings = [[0.1] * 384, [0.2] * 384]  # Mock 384-length vectors
    sample_metadatas = [
        {"source_file": "doc1.md", "title": "Pod Concepts"}, 
        {"source_file": "doc2.md", "title": "Service Concepts"}
    ]
    sample_ids = ["chunk_0", "chunk_1"]

    # 1. Add items
    db_manager.add_documents(
        model_key=test_model_key,
        texts=sample_texts,
        embeddings=sample_embeddings,
        metadatas=sample_metadatas,
        ids=sample_ids
    )

    # 2. Query items
    query_vec = [0.11] * 384
    search_results = db_manager.query_similarity(
        model_key=test_model_key, 
        query_embedding=query_vec, 
        n_results=1
    )
    
    print("\nTop Search Result:")
    print(f"Document Text: {search_results['documents'][0][0]}")
    print(f"Metadata: {search_results['metadatas'][0][0]}") 
