            
            
             ┌─────────────────┐
             │   Documents     │
             └────────┬────────┘
                      ↓
                 Extraction
                      ↓
                  Chunking
                      ↓
              ┌──────────────┐
              │  Embeddings  │
              └──────┬───────┘
                     ↓
              Vector Database
                     │
                     │
User Query ──► Query Embedding
                     │
                     ↓
              Candidate Retrieval
                     │
             ┌───────┴────────┐
             ↓                ↓
        Vector Search     Keyword/BM25
             │                │
             └───────┬────────┘
                     ↓
                  Reranker
                     ↓
              Context Selection
                     ↓
                    LLM
                     ↓
                  Response



                RAG Evaluation
88888888888888888888888888888888888888                ┌─────────────────────┐
            │ Retrieval quality   │
            ├─────────────────────┤
            │ Recall@5            │
            │ Recall@10           │
            │ MRR                 │
            └─────────────────────┘

            ┌─────────────────────┐
            │ Generation quality │
            ├─────────────────────┤
            │ Groundedness        │
            │ Answer correctness  │
            │ Citation accuracy   │
            └─────────────────────┘

            ┌─────────────────────┐
            │ System performance  │
            ├─────────────────────┤
            │ Latency             │
            │ Memory              │
            │ Cost                │
            └─────────────────────┘

____________________________

[ Ingestion Pipeline ]
  Docs Folder (.md) ──> Text Chunking ──> embedding_model ──> vector_database

[ Inference Pipeline ]
  User Query ──> integration ──> retriever ──> vector_database ──> generator ──> Final Answer
                                                       ▲
                                                app.py (UI / API)