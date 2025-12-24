import time
import streamlit as st
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from custom_llm import CustomOllamaLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Config
QDRANT_URL = "http://15.206.197.214:6333/"
QDRANT_COLLECTION = "rag_docs"
OLLAMA_BASE_URL = "https://ollama.wassan.org/api/generate"

st.set_page_config(page_title="✨ Wassan Policy Assistant", layout="centered")

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, timeout=60)

def get_embed_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

def get_llm():
    return CustomOllamaLLM(model_name="qwen2.5", base_url=OLLAMA_BASE_URL)

def load_index_from_db():
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()
    llm = get_llm()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    return index, llm

# -----------------------------
# Query UI
# -----------------------------
# Show logo image at the top 
page1, page2 = st.columns([1, 4])
with page1:

    st.image("C:/Users/WASSAN/Documents/wassan logo.jpg", width=150)
#  # adjust width as needed # Optional: add a subtitle below
with page2:
    st.markdown("### 💬 Data Roots: Wassan Policy Assistant")

    col1, col2 = st.columns([4, 1])
    with col1:
        task = st.text_area("How can I help you?", height=100, key="input")
    with col2:
        st.markdown(
            """
            <style>
            div.stButton > button {
                height: 55px;
                width: 100%;
                font-size: 18px;
                background-color: #1f77b4;
                color: white;
                border-radius: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(" ")
        send = st.button("Go✨")

    if send and task.strip():
        try:
            client = get_qdrant_client()
            count = client.count(collection_name=QDRANT_COLLECTION).count

            if count == 0:
                st.error("❌ No vector data found in DB. Please ingest documents first.")
            else:
                index, llm = load_index_from_db()
                query_engine = index.as_query_engine(llm=llm, similarity_top_k=8)
                res = query_engine.query(task)
                response_text = st.markdown(res.response)

                # placeholder = st.empty()
                # streamed = ""
                # for word in response_text.split():
                #     streamed += word + " "
                #     placeholder.markdown(f"**Assistant:** {streamed}")
                #     time.sleep(0.05)
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
