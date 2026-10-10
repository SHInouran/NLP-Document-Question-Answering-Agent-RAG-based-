from typing import Any, List, Dict
from Embedding_model import EmbeddingModel
from Vector_database import VectorDatabase


class Retriever :
    def __init__(self,db_path: str="/kaggle/working/RAG/chromaDB/test_chroma_db"):
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
    
    def reciprocal_rank_fusion(self, vector_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], k: int = 60, top_n: int = 3) -> List[Dict[str, Any]]:
        doc_scores = {}
        doc_map = {}

        # Process Vector Results
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc["text"]
            doc_map[doc_id] = doc
            score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        # Process BM25 Results
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc["text"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        # Sort merged documents by combined RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            retrieved_doc = doc_map[doc_id].copy()
            retrieved_doc["rrf_score"] = round(rrf_score, 5)
            fused_results.append(retrieved_doc)

        return fused_results

if __name__ == "__main__":
    # Test retriever logic
    print("--- Testing Retriever Module ---")

    # Initialize retriever
    retriever = Retriever(db_path="/kaggle/working/RAG/chromaDB/test_chroma_db")

    # Run a test query against our previous test database
    results = retriever.retrieve(
        query="What is a Pod in Kubernetes?", 
        model_key="minilm", 
        top_k=2
    )

    print(f"\nRetrieved {len(results)} context matches:")
    for idx, doc in enumerate(results, 1):
        print(f"\nResult #{idx} (Score: {doc['score']}):")
        print(f"Source: {doc['metadata'].get('source')}")
        print(f"Title:  {doc['metadata'].get('title')}")
        print(f"Text:   {doc['text']}")