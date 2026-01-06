import time
import streamlit as st
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from custom_llm import CustomOllamaLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Config
QDRANT_URL = "http://15.206.197.214:6333/"
QDRANT_COLLECTION = "rag_docs"
OLLAMA_BASE_URL = "https://ollama.wassan.org/api/generate"

   

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
st.set_page_config(page_title="✨ Wassan Policy Assistant", layout="centered")
 # local file in your project folder 
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0.1rem;
        }
        .stTextInput, .stSelectbox, .stButton {
            margin-bottom: 0.2rem;
        }
    </style>
""", unsafe_allow_html=True)
page1, page2 = st.columns([1, 4])
with page1:
    st.markdown(" ")
    st.image("./media/wassan_logo.png", width=70)
#  # adjust width as needed # Optional: add a subtitle below
with page2:
    st.markdown(" ")
    st.markdown("<h2 style='color:#1f77b4;'>Wassan Internal Policy Assistant</h2>", unsafe_allow_html=True)
    # st.title("✨ WASSAN Policy Assistant")
    st.caption("Your interactive guide to organizational HR & other policies, leave rules, benefits and procedures.")

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
                # res = query_engine.query(task)
                with st.spinner('Generating..'):
                    res = query_engine.query(task)

                if not res.response.strip():
                    st.warning("I couldn’t find a clear answer. \
                                For further information, please reach out to hr@wassan.org")
                else:
                    
                    st.markdown(f"**Assistant:** {res.response}")
                    st.markdown("For the correctness of the information, Please check with the HR team.")
                    st.markdown(" ")
                    st.markdown(" ")
                    st.markdown(" ")
                    st.markdown(" ")
                    st.markdown(" ")
                
        except Exception as e:
            st.error(f"❌ Query failed: {e}")




import base64


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = image_to_base64("media/logo v2.png")

st.markdown(
    f"""
    <style>
        .footer {{
            position: fixed;
            right: 0;              /* 👈 align to right */
            bottom: 0;             /* 👈 stick to bottom */
            width:100%;
            background-color: #f0f8ff;
            text-align: right;     /* 👈 text aligned right */
            padding: 10px;
            font-size: 14px;
            color: grey;
            # border-top-left-radius: 8px; /* rounded corner look */
        }}
        # .footer img {{
        #     display: block;
        #     margin-left: auto;     /* 👈 push image to right */
        # }}
    </style>
    <div class="footer">
        <img src="data:image/png;base64,{logo_b64}" width="70">
        Developed by <b>Data Roots Team</b>
    </div>
    """,
    unsafe_allow_html=True
)