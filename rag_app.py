import os
import streamlit as st
import chromadb
import tempfile
from pathlib import Path
from groq import Groq
from PyPDF2 import PdfReader

# ─── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="📚 My RAG App",
    page_icon="📚",
    layout="wide"
)

# ─── CUSTOM STYLING ───────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
    }

    .main {
        background: #0f0f13;
        color: #e8e6e0;
    }

    .stApp {
        background: #0f0f13;
    }

    section[data-testid="stSidebar"] {
        background: #16161d !important;
        border-right: 1px solid #2a2a35;
    }

    .upload-box {
        border: 2px dashed #3d3d52;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #16161d;
        transition: border-color 0.2s;
    }

    .doc-chip {
        display: inline-block;
        background: #1e1e2e;
        border: 1px solid #3d3d52;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 13px;
        margin: 4px 4px 4px 0;
        color: #a0a0c0;
    }

    .chat-user {
        background: #1a1a28;
        border-left: 3px solid #7c6af7;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #e8e6e0;
    }

    .chat-bot {
        background: #161622;
        border-left: 3px solid #4ade80;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #e8e6e0;
    }

    .stat-box {
        background: #16161d;
        border: 1px solid #2a2a35;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .stTextInput > div > div > input {
        background: #16161d !important;
        border: 1px solid #3d3d52 !important;
        color: #e8e6e0 !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background: #7c6af7 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    div[data-testid="stFileUploader"] {
        background: #16161d;
        border: 2px dashed #3d3d52;
        border-radius: 12px;
        padding: 1rem;
    }

    .stSelectbox > div > div {
        background: #16161d !important;
        border: 1px solid #3d3d52 !important;
        color: #e8e6e0 !important;
    }

    hr {
        border-color: #2a2a35 !important;
    }

    .stSuccess {
        background: #0d2318 !important;
        color: #4ade80 !important;
    }

    .stWarning {
        background: #1e180a !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── CONFIG ───────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 800

# ─── INIT DATABASE ────────────────────────────────────────
@st.cache_resource
def get_collection():
    client_db = chromadb.PersistentClient(path="./chroma_db")
    return client_db.get_or_create_collection(name="rag_docs")

collection = get_collection()

# ─── SESSION STATE ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ─── HELPER: EXTRACT TEXT ─────────────────────────────────
def extract_text(file) -> str:
    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    elif name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    else:
        return ""

# ─── HELPER: STORE CHUNKS ─────────────────────────────────
def store_document(text: str, filename: str) -> int:
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    chunks = [c.strip() for c in chunks if c.strip()]

    existing_ids = set(collection.get()["ids"])
    new_texts, new_ids = [], []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}"
        if chunk_id not in existing_ids:
            new_texts.append(chunk)
            new_ids.append(chunk_id)

    if new_texts:
        collection.add(documents=new_texts, ids=new_ids)

    return len(new_texts)

# ─── HELPER: SEARCH ───────────────────────────────────────
def search_docs(question: str, n: int = 4) -> str:
    total = collection.count()
    if total == 0:
        return ""
    results = collection.query(
        query_texts=[question],
        n_results=min(n, total)
    )
    return "\n\n---\n\n".join(results["documents"][0])

# ─── HELPER: ASK GROQ ─────────────────────────────────────
def ask_groq(question: str, api_key: str) -> str:
    context = search_docs(question)

    if not context:
        return "⚠️ No documents uploaded yet. Please upload a file first!"

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the uploaded documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2
    )
    return response.choices[0].message.content

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown("---")

    # API Key input
    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com"
    )

    st.markdown("---")
    st.markdown("## 📁 Upload Documents")

    # File uploader — supports PDF and TXT
    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded:
        for file in uploaded:
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    text = extract_text(file)
                    if text.strip():
                        count = store_document(text, file.name)
                        st.session_state.uploaded_files.append(file.name)
                        st.success(f"✅ {file.name} — {count} chunks stored!")
                    else:
                        st.warning(f"⚠️ Could not read {file.name}")

    st.markdown("---")

    # Uploaded files list
    if st.session_state.uploaded_files:
        st.markdown("**📚 Loaded documents:**")
        for fname in st.session_state.uploaded_files:
            st.markdown(f"<span class='doc-chip'>📄 {fname}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Stats
    total_chunks = collection.count()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='stat-box'><div style='font-size:22px;font-weight:700;color:#7c6af7'>{len(st.session_state.uploaded_files)}</div><div style='font-size:11px;color:#606080'>files</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-box'><div style='font-size:22px;font-weight:700;color:#4ade80'>{total_chunks}</div><div style='font-size:11px;color:#606080'>chunks</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Clear database button
    if st.button("🗑️ Clear All Documents"):
        client_db = chromadb.PersistentClient(path="./chroma_db")
        client_db.delete_collection("rag_docs")
        st.session_state.uploaded_files = []
        st.session_state.messages = []
        st.rerun()

# ══════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════
st.markdown("# 📚 RAG Chat")
st.markdown("Upload any PDF or text file, then ask questions about it.")
st.markdown("---")

# Chat history display
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>🧑 <b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>🤖 <b>Groq:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Question input
st.markdown("### 💬 Ask a question")
col_input, col_btn = st.columns([5, 1])

with col_input:
    question = st.text_input(
        "question",
        placeholder="e.g. What is this book about? Who is the main character?",
        label_visibility="collapsed"
    )

with col_btn:
    send = st.button("Send ➤")

# Handle send
if send and question:
    if not api_key:
        st.error("❌ Please enter your Groq API key in the sidebar!")
    elif collection.count() == 0:
        st.warning("⚠️ Please upload a document first!")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🔍 Searching your documents..."):
            answer = ask_groq(question, api_key)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# Empty state
if collection.count() == 0:
    st.markdown("""
    <div style='text-align:center; padding: 3rem; color: #606080;'>
        <div style='font-size: 48px;'>📂</div>
        <div style='font-size: 18px; margin-top: 1rem; font-family: Syne, sans-serif;'>No documents loaded yet</div>
        <div style='font-size: 14px; margin-top: 0.5rem;'>Upload a PDF or TXT file from the sidebar to get started</div>
    </div>
    """, unsafe_allow_html=True)