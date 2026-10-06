import os
from typing import List, Dict, Any
from openai import OpenAI

#in terminal:

class AnswerGenerator:
    def __init__(
        self, 
        # Recommended free production model on Groq
        model_name: str = "openai/gpt-oss-120b", 
        api_key: str = None,
        base_url: str = "https://api.groq.com/openai/v1"
    ):
        """
        Initializes the LLM generator configured for Groq API.
        """
        # Read GROQ_API_KEY from environment if not passed explicitly
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "API Key missing! Set the 'GROQ_API_KEY' environment variable "
                "or pass `api_key` directly to AnswerGenerator."
            )

        self.model_name = model_name
        
        # Point the OpenAI client to Groq's API endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

    def _build_context_string(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved doc chunks into a numbered context block."""
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source', 'Unknown Source')
            title = doc['metadata'].get('title', 'Untitled Section')
            text = doc['text']
            
            context_block = (
                f"--- [SOURCE {idx}]: {title} ({source}) ---\n"
                f"{text}\n"
            )
            context_parts.append(context_block)
            
        return "\n".join(context_parts)

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combines context and query, calling Groq to generate a response."""
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant Kubernetes documentation to answer your question.",
                "sources": []
            }

        formatted_context = self._build_context_string(retrieved_docs)

        system_prompt = (
            "You are an expert technical assistant specialized in Kubernetes documentation.\n"
            "Your task is to answer the user's question strictly using the provided context blocks.\n\n"
            "Rules:\n"
            "1. Rely ONLY on the information presented in the CONTEXT below. Do NOT invent facts.\n"
            "2. If the context does not contain enough information to answer, state clearly that "
            "you do not know based on the provided docs.\n"
            "3. Reference source titles or source numbers where appropriate."
        )

        user_prompt = (
            f"CONTEXT:\n{formatted_context}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            
            answer_text = response.choices[0].message.content

        except Exception as e:
            answer_text = f"Error generating answer from Groq: {str(e)}"

        sources = [
            {
                "title": doc['metadata'].get('title', 'Unknown'),
                "source": doc['metadata'].get('source', 'Unknown')
            }
            for doc in retrieved_docs
        ]

        return {
            "answer": answer_text,
            "sources": sources
        }


if __name__ == "__main__":
    print("--- Testing Generator Module with Groq ---")

    # Initialize generator using default Groq settings
    generator = AnswerGenerator()
    
    test_query = "What is a Pod in Kubernetes?"
    result = generator.generate_answer(query=test_query, retrieved_docs=mock_docs)

    print(f"\nUser Question: {test_query}")
    print(f"\nGenerated Answer:\n{result['answer']}")
    print(f"\nSources cited: {result['sources']}")