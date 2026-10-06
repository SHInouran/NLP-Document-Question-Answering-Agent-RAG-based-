import os
from typing import List, Dict, Any
from Retriever import Retriever
from Generator import AnswerGenerator

class RAGPipeline:
    def __init__(self, db_path:str ="/kaggle/working/RAG/chroma_db", llm_model:str ="openai/gpt-oss-120b" , groq_api_key:str = None):
        self.retriever = Retriever(db_path=db_path)
        self.answer_generator = AnswerGenerator(model_name=llm_model, api_key=groq_api_key)

    def Query( self, query: str, model_key: str ="minilm", top_k: int = 3) ->Dict[str,Any]:
        print(f"question:{query}\nEmbedding model:{model_key}\nRetrieving top {top_k} relevant documents...")
        retrieved_docs = self.retriever.retrieve(query=query, model_key=model_key, top_k=top_k)
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
    pipeline = RAGPipeline(db_path="/kaggle/working/RAG/chromaDB/test_real_db")

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