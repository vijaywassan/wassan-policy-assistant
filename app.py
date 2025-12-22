import os
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext
)
# import chromadb
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.vector_stores.chroma import ChromaVectorStore
from custom_llm import CustomOllamaLLM

DOCS_DIR = "./docs"
# CHROMA_PATH = "./vector_store"

# ✅ Ensure docs folder exists
os.makedirs(DOCS_DIR, exist_ok=True)

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def build_index():
    try:
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    except ValueError as e:
        if "No files found" in str(e):
            return None, None, None
        else:
            raise e

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    llm = CustomOllamaLLM(
        model_name="llama3",
        base_url="https://ollama.wassan.org/api/generate"
    )
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ Connect to your cloud Qdrant instance
    qdrant_client = QdrantClient(
        url="http://15.206.197.214:6333/",  # replace with your actual URL
        timeout=60  # seconds
        # api_key="Qdrant@123"                 # if authentication is enabled
    )
    

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="rag_docs"
    )
    

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )

    return index, llm, embed_model
# Sidebar section to browse chunks

# st.text('test', build_index.qdrant_client.get_collections())
# st.title("Wassan Policy Assistant")
##########################################
#ui
st.set_page_config(page_title="✨ Wassan policy assistant(V1)", layout="centered")
import os, json
# --- Simple Admin Credentials ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "secret123"    

# --- Initialize session state ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# --- Sidebar UI ---
st.sidebar.title("🔑 Admin Panel")

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

    # ✅ Upload section
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents to add to RAG",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            save_path = os.path.join(DOCS_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        st.sidebar.success("✅ Files uploaded successfully!")
    
        if st.sidebar.button("Rebuild Index"):
            index, llm, embed_model = build_index()
            st.sidebar.success("✅ Index rebuilt with new documents!")

# ✅ Query section
# import streamlit as st
import streamlit as st
import time
st.title("💬 Wassan Policy Assistant")

# st.title("💬 Wassan Policy Assistant")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# # Display previous messages (thread style)
# for msg in st.session_state["messages"]:
#     if msg["role"] == "user":
#         st.markdown(f"**You:** {msg['content']}")
#     else:
#         st.markdown(f"**Assistant:** {msg['content']}")

# Create two columns: text area (wide) and button (narrow)
col1, col2 = st.columns([4,1])

with col1:
    task = st.text_area("How can I help you?", height=100, key="input")

with col2:
    st.markdown("""
        <style>
        div.stButton > button {
            height: 55px;          /* Match text_area height */
            width: 100%;           /* Stretch to column width */
            font-size: 18px;       /* Bigger text */
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(" ")
    send = st.button("Go✨")

if send and task.strip():    
        # Build index and query
        index, llm, embed_model = build_index()        
        # from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(url="http://15.206.197.214:6333")
        count = qdrant_client.count(collection_name="rag_docs").count
        if count == 0: 
            st.error("❌ No vector data found in DB. Please ingest documents first.")
        else:
            query_engine = index.as_query_engine(llm=llm)
            res = query_engine.query(task)
            response_text = res.response
    
            # Stream assistant response word by word
            placeholder = st.empty()
            streamed = ""
            for word in response_text.split():
                streamed += word + " "
                placeholder.markdown(f"**Assistant:** {streamed}")
                time.sleep(0.05)

#important commands/instructions/steps to smooth run and fast. 
#curl -X DELETE "http://15.206.197.214:6333/collections/rag_docs"
#above command to remove content from the db
