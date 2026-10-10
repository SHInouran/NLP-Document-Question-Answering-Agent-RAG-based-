import argparse
import os

from Embedding_model import EmbeddingModel
from Retriever import Retriever
from Vector_database import VectorDatabase
from bm25_retriever import BM25Retriever
from ingestion import ingest_docs


def main():
    parser = argparse.ArgumentParser(description="Diagnose document loading and retrieval.")
    parser.add_argument("--docs-dir", default="/kaggle/working/RAG/documents")
    parser.add_argument("--db-path", default="/kaggle/working/RAG/chromaDB/real_test_db")
    parser.add_argument("--model-key", default="minilm")
    parser.add_argument("--query", default="What is a Pod in Kubernetes?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"docs_dir exists: {os.path.exists(args.docs_dir)} -> {args.docs_dir}")
    docs = ingest_docs(args.docs_dir) if os.path.exists(args.docs_dir) else []
    print(f"markdown chunks loaded: {len(docs)}")

    db = VectorDatabase(db_path=args.db_path)
    print(f"collection k8s_docs_{args.model_key} count: {db.collection_count(args.model_key)}")

    if docs:
        bm25_results = BM25Retriever(docs).retrieve(args.query, top_k=args.top_k)
        print(f"BM25 results: {len(bm25_results)}")
        for idx, doc in enumerate(bm25_results[:3], 1):
            print(f"BM25 #{idx}: score={doc['bm25_score']:.4f} source={doc['metadata'].get('source')}")

    try:
        vector_results = Retriever(db_path=args.db_path).retrieve(
            query=args.query,
            model_key=args.model_key,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"Vector retrieval error: {exc}")
        return

    print(f"Vector results: {len(vector_results)}")
    for idx, doc in enumerate(vector_results[:3], 1):
        print(f"Vector #{idx}: score={doc['score']} source={doc['metadata'].get('source')}")
        print(doc["text"][:250].replace("\n", " "))


if __name__ == "__main__":
    main()
