import chromadb
from typing import Any, List,Dict
from sentence_transformers import SentenceTransformer

class VectorDatabase:
    def __init__(self, db_path:str = "/kaggle/working/RAG/chromaDB/test_chroma_db"):
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

        results = collection.query(query_embeddings = [query_embedding], n_results = n_results)
        return results

if __name__ == "__main__":
    print("--- Testing Vector Database Manager with Real Embeddings ---")

    # 1. Initialize DB Manager
    test_db_path = "/kaggle/working/RAG/chromaDB/test_real_db"
    db_manager = VectorDatabase(db_path=test_db_path)
    
    test_model_key = "minilm"
    print(f"Loading local embedding model for testing...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 2. Actual Kubernetes documentation chunks & metadata
    sample_texts = [
        "A Pod is the smallest execution unit in Kubernetes. A Pod encapsulates one or more applications.",
        "Services expose an application running on a set of Pods as a network service in Kubernetes.",
        "Deployments manage stateless applications and provide declarative updates for Pods and ReplicaSets.",
        "ConfigMaps allow you to decouple configuration artifacts from image content to keep containerized applications portable."
    ]
    
    sample_metadatas = [
        {"source_file": "docs/concepts/workloads/pods.md", "title": "Pod Concepts", "category": "workloads"},
        {"source_file": "docs/concepts/services-networking/service.md", "title": "Service Concepts", "category": "networking"},
        {"source_file": "docs/concepts/workloads/controllers/deployment.md", "title": "Deployments", "category": "workloads"},
        {"source_file": "docs/concepts/configuration/configmap.md", "title": "ConfigMaps", "category": "configuration"}
    ]
    
    sample_ids = [f"k8s_chunk_{i}" for i in range(len(sample_texts))]

    # 3. Generate real 384-dimensional embeddings
    print("Generating real embeddings for sample documents...")
    real_embeddings = embedder.encode(sample_texts, convert_to_numpy=True).tolist()

    # 4. Insert into ChromaDB
    db_manager.add_documents(
        model_key=test_model_key,
        texts=sample_texts,
        embeddings=real_embeddings,
        metadatas=sample_metadatas,
        ids=sample_ids
    )

    # 5. Query using an actual query string
    test_query = "How do I expose my app to network traffic?"
    print(f"\nQuerying DB for: '{test_query}'")
    
    query_vector = embedder.encode([test_query], convert_to_numpy=True).tolist()[0]
    search_results = db_manager.query_similarity(
        model_key=test_model_key, 
        query_embedding=query_vector, 
        n_results=2
    )

    # 6. Print top search results
    print("\n--- Search Results ---")
    for idx, (doc, meta, dist) in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0], search_results['distances'][0]), 1):
        print(f"\nResult #{idx} (Distance: {dist:.4f}):")
        print(f"Title: {meta.get('title')}")
        print(f"Source: {meta.get('source_file')}")
        print(f"Content: {doc}")