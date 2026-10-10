import argparse
import os
from typing import List

from Embedding_model import EmbeddingModel
from Vector_database import VectorDatabase
from ingestion import ingest_docs


def batched(items: List[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def main():
    parser = argparse.ArgumentParser(description="Build the Chroma vector index for the RAG app.")
    parser.add_argument("--docs-dir", default="/kaggle/working/RAG/documents")
    parser.add_argument("--db-path", default="/kaggle/working/RAG/chromaDB/real_test_db")
    parser.add_argument("--model-key", default="minilm")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--id-prefix", default="k8s")
    args = parser.parse_args()

    if not os.path.exists(args.docs_dir):
        raise FileNotFoundError(f"Documents directory not found: {args.docs_dir}")

    docs = ingest_docs(args.docs_dir)
    if not docs:
        raise ValueError(f"No markdown chunks found in {args.docs_dir}")

    embedder = EmbeddingModel(model_key=args.model_key)
    db = VectorDatabase(db_path=args.db_path)

    print(f"Loaded {len(docs)} chunks from {args.docs_dir}")
    print(f"Building collection k8s_docs_{args.model_key} in {args.db_path}")

    for start, batch in batched(docs, args.batch_size):
        texts = [doc["text"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        ids = [f"{args.id_prefix}_{args.model_key}_{start + i}" for i in range(len(batch))]
        embeddings = embedder.embedd_batch(texts, batch_size=args.batch_size)
        db.add_documents(
            model_key=args.model_key,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Indexed {min(start + len(batch), len(docs))}/{len(docs)} chunks")

    print(f"Final collection count: {db.collection_count(args.model_key)}")


if __name__ == "__main__":
    main()
