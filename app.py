import os
import time
import streamlit as st
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from custom_llm import CustomOllamaLLM


# -----------------------------
# Config
# -----------------------------
DOCS_DIR = "./docs"
QDRANT_URL = "http://15.206.197.214:6333/"
QDRANT_COLLECTION = "rag_docs"
OLLAMA_BASE_URL = "https://ollama.wassan.org/api/generate"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "secret123"

os.makedirs(DOCS_DIR, exist_ok=True)
st.set_page_config(page_title="✨ Wassan policy assistant (V2)", layout="centered")


# -----------------------------
# Helpers
# -----------------------------
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, timeout=60)


def get_embed_model():
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm():
    return CustomOllamaLLM(model_name="llama3", base_url=OLLAMA_BASE_URL)


def build_index_from_db():
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


def rebuild_index_from_docs(uploaded_files):
    for file in uploaded_files:
        save_path = os.path.join(DOCS_DIR, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    nodes = SimpleNodeParser().get_nodes_from_documents(documents)

    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = get_embed_model()

    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    count = client.count(collection_name=QDRANT_COLLECTION).count
    return count


def show_db_status():
    client = get_qdrant_client()
    try:
        count = client.count(collection_name=QDRANT_COLLECTION).count
        st.sidebar.info(f"📊 Current vector count in DB: {count}")
    except Exception:
        st.sidebar.error("⚠️ Could not connect to Qdrant. Check URL/network.")


# -----------------------------
# Admin panel
# -----------------------------
st.sidebar.title("🔑 Admin Panel")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login", use_container_width=True):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state["authenticated"] = True
            st.sidebar.success("✅ Logged in as Admin")
        else:
            st.sidebar.error("❌ Invalid credentials")
else:
    st.sidebar.success("✅ Logged in as Admin")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.sidebar.info("🔒 Logged out")

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents to add to RAG",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.sidebar.button("Rebuild Index"):
            try:
                count = rebuild_index_from_docs(uploaded_files)
                st.sidebar.success("✅ Index rebuilt with new documents!")
                st.sidebar.info(f"📊 Current vector count in DB: {count}")
            except Exception as e:
                st.sidebar.error(f"❌ Rebuild failed: {e}")

# Always show DB status
show_db_status()


# -----------------------------
# Query UI
# -----------------------------
st.title("💬 Wassan Policy Assistant")

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
            index, llm = build_index_from_db()
            query_engine = index.as_query_engine(llm=llm)
            res = query_engine.query(task)
            response_text = res.response

            placeholder = st.empty()
            streamed = ""
            for word in response_text.split():
                streamed += word + " "
                placeholder.markdown(f"**Assistant:** {streamed}")
                time.sleep(0.05)
    except Exception as e:
        st.error(f"❌ Query failed: {e}")
#Vector storage ≈ 1440 × 1.5 KB ≈ 2.1 MB.