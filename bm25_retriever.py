import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, docs: List[Dict[str, Any]]):
        """
        docs: List of dicts containing 'text' and 'metadata' from ingestion.py
        """
        self.docs = docs
        self.corpus = [doc["text"] for doc in docs]
        
        # Tokenize corpus (simple lowercase word splitting)
        self.tokenized_corpus = [self._tokenize(doc_text) for doc_text in self.corpus]
        
        # Build BM25 Index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and keep alphanumeric tokens/hyphenated words (e.g., k8s terms like "kubectl")
        return re.findall(r'\b\w+(?:-\w+)*\b', text.lower())

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Rank document indices by BM25 score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return documents with matching terms
                doc = self.docs[idx].copy()
                doc["bm25_score"] = float(scores[idx])
                results.append(doc)
                
        return results