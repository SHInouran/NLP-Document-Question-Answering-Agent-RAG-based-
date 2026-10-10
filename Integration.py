import os
from typing import List, Dict, Any
from Retriever import Retriever
from Generator import AnswerGenerator
from bm25_retriever import BM25Retriever
from ingestion import ingest_docs

class RAGPipeline:
    def __init__(self, db_path:str ="/kaggle/working/RAG/chromaDB/real_test_db", llm_model:str ="openai/gpt-oss-120b" , groq_api_key:str = None ):
        self.docs = ingest_docs(docs_dir="/kaggle/working/RAG/documents") 
        self.retriever = Retriever(db_path=db_path)
        self.bm25 = BM25Retriever(db_path= db_path)
        self.answer_generator = AnswerGenerator(model_name=llm_model, api_key=groq_api_key)
        


    def Query( self, query: str, model_key: str ="minilm", top_k: int = 30) ->Dict[str,Any]:
        print(f"question:{query}\nEmbedding model:{model_key}\nRetrieving top {top_k} relevant documents...")
        vector_retrieved_docs = self.retriever.retrieve(query=query, model_key=model_key, top_k=top_k)
        bm_retrieved_docs = self.bm25.retrieve(query=query , top_k=top_k)
        retrieved_docs = self.retriever.reciprocal_rank_fusion( vector_results= vector_retrieved_docs , bm25_results= bm_retrieved_docs , k= 50 , top_n= 10)
        print(f"Retrieved {len(retrieved_docs)} documents. Generating answer...")
        answer_response = self.answer_generator.generate_answer(query=query, retrieved_docs=retrieved_docs)
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "answer": answer_response.get("answer"),
            "sources": answer_response.get("sources")
        }

if __name__ == "__main__":
    
    # Ensure GROQ_API_KEY environment variable is set
    if not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = "gsk_your_actual_groq_key_here"

    print("--- Testing Full RAG Pipeline ---")

    # Initialize overall pipeline (connected to our test vector DB)
    pipeline = RAGPipeline(db_path="/kaggle/working/RAG/chromaDB/real_test_db")

    # Run query end-to-end
    result = pipeline.Query(
        query="What is a Pod in Kubernetes?",
        model_key="minilm",
        top_k=2
    )

    print("\n================ FINAL RESPONSE ================")
    print(f"ANSWER:\n{result['answer']}\n")
    print("CITED SOURCES:")
    for source in result['sources']:
        print(f" - {source['title']} ({source['source']})")