import numpy as np
from typing import List, Dict, Any

class RetrievalEvaluator:
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        top_k_retrieved = retrieved_ids[:k]
        relevant_retrieved = set(top_k_retrieved).intersection(set(ground_truth_ids))
        return len(relevant_retrieved) / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        top_k_retrieved = retrieved_ids[:k]
        relevant_retrieved = set(top_k_retrieved).intersection(set(ground_truth_ids))
        return len(relevant_retrieved) / len(ground_truth_ids) if ground_truth_ids else 0.0

    @staticmethod
    def mrr_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in ground_truth_ids:
                return 1.0 / rank
        return 0.0


# --- Example Evaluation Benchmark Run ---
if __name__ == "__main__":
    # 1. Define Ground Truth Test Dataset
    test_cases = [
        {
            "query": "What is a Pod in Kubernetes?",
            "ground_truth_ids": ["k8s_chunk_0"]  # Expected chunk ID in vector DB
        },
        {
            "query": "How do I expose my app to network traffic?",
            "ground_truth_ids": ["k8s_chunk_1"]
        }
    ]

    # 2. Run test queries through your hybrid retriever
    # (Assuming retriever.retrieve returns documents with 'id' or 'source' in metadata)
    evaluator = RetrievalEvaluator()
    k = 3

    p_scores, r_scores, mrr_scores = [], [], []

    for test in test_cases:
        # Mocking retrieved IDs from your pipeline query output
        # In actual usage: retrieved_docs = retriever.retrieve(test["query"], top_k=k)
        retrieved_ids = ["k8s_chunk_0", "k8s_chunk_3", "k8s_chunk_1"] 

        p_scores.append(evaluator.precision_at_k(retrieved_ids, test["ground_truth_ids"], k))
        r_scores.append(evaluator.recall_at_k(retrieved_ids, test["ground_truth_ids"], k))
        mrr_scores.append(evaluator.mrr_at_k(retrieved_ids, test["ground_truth_ids"], k))

    print("--- Retrieval Evaluation Metrics ---")
    print(f"Mean Precision@{k}: {np.mean(p_scores):.4f}")
    print(f"Mean Recall@{k}:    {np.mean(r_scores):.4f}")
    print(f"Mean MRR@{k}:       {np.mean(mrr_scores):.4f}")