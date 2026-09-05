# importing

from PyPDF2 import PdfReader
import streamlit as st
from groq import Groq
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration

MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Extraction of text from PDF and TXT

def extract_text(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    elif name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    return ""

# Chunking with LangChain

def chunk_text(text: str) -> list[str]:
    """Split raw text into overlapping, semantically-aware chunks."""
    chunks = text_splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]

# Storing chunks in ChromaDB

@st.cache_resource
def get_collection():
    client_db = chromadb.PersistentClient(path="./chroma_db")
    return client_db.get_or_create_collection(name="rag_docs")

collection = get_collection()

def store_document(text: str, filename: str) -> int:
    """Chunk a document and store new chunks in the vector DB."""
    chunks = chunk_text(text)

    existing_ids = set(collection.get()["ids"])
    new_docs, new_ids = [], []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}"
        if chunk_id not in existing_ids:
            new_docs.append(chunk)
            new_ids.append(chunk_id)

    if new_docs:
        collection.add(documents=new_docs, ids=new_ids)

    return len(new_docs)

# Retrieval — Searching stored chunks

def search_docs(question: str, n: int = 4) -> str:
    """Find the most relevant stored chunks for a given question."""
    total = collection.count()
    if total == 0:
        return ""

    results = collection.query(query_texts=[question], n_results=min(n, total))
    return "\n\n---\n\n".join(results["documents"][0])

# Generation — Calling Groq with the retrieved context

def ask_groq(question: str, api_key: str) -> str:
    """Retrieve relevant context, then ask the LLM to answer using only that context."""
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

# User interface

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    st.markdown("## ⚙️ Setup")
    api_key = st.secrets["GROQ_API_KEY"]

    st.markdown("## 📁 Upload Documents")
    uploaded = st.file_uploader("Drop files here", type=["pdf", "txt"], accept_multiple_files=True)

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

st.markdown("# 📚 RAG Chat")

for msg in st.session_state.messages:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Groq"
    st.markdown(f"**{role}:** {msg['content']}")

question = st.text_input("Ask a question", placeholder="What is this document about?")
send = st.button("Send ➤")

if send and question:
    if not api_key:
        st.error("❌ Please enter your Groq API key!")
    elif collection.count() == 0:
        st.warning("⚠️ Please upload a document first!")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("🔍 Searching your documents..."):
            answer = ask_groq(question, api_key)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
