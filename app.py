import os
import streamlit as st
from Integration import RAGPipeline
from Embedding_model import EmbeddingModel

# Page Configuration
st.set_page_config(
    page_title="Kubernetes Local RAG Assistant",
    page_icon="☸️",
    layout="wide"
)

st.title("☸️ Local Kubernetes Documentation Assistant")
st.markdown("Ask technical questions against official Kubernetes documentation using local embeddings and Groq LLM.")

# 1. Sidebar Configuration
st.sidebar.header("⚙️ Pipeline Configuration")

# Groq API Key Input
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password",
    help="Get a free key from https://console.groq.com"
)

# Embedding Model Selection
model_options = list(EmbeddingModel.MODELS.keys())
selected_model_key = st.sidebar.selectbox(
    "Select Embedding Model",
    options=model_options,
    index=0,
    help="Choose the model used to embed queries and search the vector database."
)

# Show selected model parameters
model_info = EmbeddingModel.MODELS[selected_model_key]
st.sidebar.caption(f"**Repo:** `{model_info['name']}`")
st.sidebar.caption(f"**Dimensions:** `{model_info['dim']}`")
st.sidebar.caption(f"**Description:** {model_info['description']}")

st.sidebar.divider()

# Retrieval parameters
top_k = st.sidebar.slider(
    "Top K Retrieved Documents",
    min_value=1,
    max_value=10,
    value=4,
    help="Number of document chunks passed to the LLM as context."
)

# 2. Pipeline Initialization (Cached for performance)
@st.cache_resource(show_spinner=False)
def load_pipeline(api_key: str):
    db_path = "/kaggle/working/RAG/chroma_db" if os.path.exists("/kaggle/working/RAG") else "./chroma_db"
    return RAGPipeline(
        db_path=db_path,
        llm_model="openai/gpt-oss-120b",
        groq_api_key=api_key
    )

# Validate API Key
if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to run the pipeline.")
    st.stop()

try:
    pipeline = load_pipeline(groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline: {e}")
    st.stop()

# 3. Main Query Interface
query = st.text_input(
    "Enter your Kubernetes question:",
    placeholder="How do I expose a deployment using a Service?"
)

if st.button("Submit Question", type="primary"):
    if not query.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Embedding query, searching ChromaDB, and generating response..."):
            # Execute RAG query
            result = pipeline.Query(
                query=query,
                model_key=selected_model_key,
                top_k=top_k
            )

        # Display Answer
        st.subheader("💡 Answer")
        st.markdown(result["answer"])

        st.divider()

        # Display Sources and Retrieved Context
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📚 Cited Sources")
            if result["sources"]:
                for idx, src in enumerate(result["sources"], 1):
                    st.markdown(f"**{idx}. {src['title']}**")
                    st.caption(f"Path: `{src['source_file']}`")
            else:
                st.write("No sources cited.")

        with col2:
            st.subheader("🔍 Context Details")
            for idx, doc in enumerate(result["retrieved_documents"], 1):
                with st.expander(f"Chunk #{idx} | Similarity Score: {doc['score']}"):
                    st.markdown(f"**Source File:** `{doc['metadata'].get('source')}`")
                    st.text_area("Content", value=doc["text"], height=150, key=f"chunk_{idx}")