# 📚 DocuMind — RAG-Powered Knowledge Assistant

### Ask Your Documents Anything, Get Answers Grounded in Reality


![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![LangChain](https://img.shields.io/badge/LangChain-Text%20Splitting-1C3C3C?logo=langchain) ![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange) ![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-F55036) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that turns any PDF or text document into an interactive knowledge base — answering questions using only what's actually in your files, not the model's guesses.

---

## The Problem

Large Language Models are confident — even when they're wrong. Ask a raw LLM a question about a specific document, and it will either refuse, or worse, **hallucinate** a plausible-sounding answer that has nothing to do with your actual content.

DocuMind solves this by **retrieving the exact relevant passages** from your uploaded documents before the LLM ever generates a response — grounding every answer in real, traceable source text, and explicitly admitting when it doesn't know.

---

## Pipeline Overview

Uploaded PDF / TXT
│
▼
Text Extraction
(PyPDF2 — page-by-page, skips unreadable pages)
│
▼
Semantic Chunking
(LangChain Recursive Splitter — 500 tokens, 50-token overlap)
│
▼
Vector Embedding + Storage
(ChromaDB — persistent, deduplicated)
│
▼
Similarity Search
(Cosine similarity — top-4 relevant chunks retrieved)
│
▼
Grounded Generation
(Groq LPU Inference — context-restricted prompt)
│
▼
Streamlit Chat Interface

---

## Tech Stack

| Layer | Tools |
|---|---|
| Text Extraction | PyPDF2 |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (tiktoken-based, token-accurate) |
| Vector Database | ChromaDB (`PersistentClient`, cosine similarity search) |
| Embeddings | ChromaDB default (`all-MiniLM-L6-v2` sentence-transformer) |
| LLM Inference | Groq API (`openai/gpt-oss-20b`) |
| Interface | Streamlit |
| Secrets Management | Streamlit `secrets.toml` |

---

## How Retrieval Stays Accurate

Every question passes through a grounding pipeline before the LLM ever sees it, to keep answers tied to real document content instead of the model's training data.

User Question
│
▼
Embed Question
(same embedding model as stored chunks)
│
▼
Cosine Similarity Search
(top-4 most relevant chunks pulled from ChromaDB)
│
▼
Context-Restricted Prompt
("Answer using ONLY the context below")
│
▼
Explicit Refusal Fallback
("I don't have that information" — if answer isn't in context)
│
▼
Final Answer


**Why the refusal fallback matters:** without it, LLMs default to sounding helpful even when they're guessing. Giving the model explicit permission to say "I don't know" is what keeps answers trustworthy rather than just plausible.

---

## Design Decisions Worth Knowing

| Decision | Why |
|---|---|
| **500-token chunks, 50-token overlap** | Small enough to stay topically focused for retrieval, with overlap preventing key sentences from being severed at chunk boundaries |
| **Recursive (not fixed-size) chunking** | Splits on paragraph → sentence → word boundaries first, avoiding mid-sentence cuts that plain character-slicing causes |
| **Deterministic chunk IDs (`filename_index`)** | Re-uploading the same file never creates duplicate vectors in the database |
| **Groq for inference** | Sub-second response times thanks to LPU hardware, keeping the chat interface feeling instant rather than sluggish |
| **Low temperature (0.2)** | Prioritizes consistent, factual answers over creative variation — appropriate for document Q&A |

---

## Dashboard Structure

| Section | Description |
|---|---|
| **📁 Upload Documents** | Drag-and-drop PDF/TXT ingestion with live chunk-count feedback per file |
| **💬 Ask a Question** | Natural language query box, answered strictly from retrieved document context |
| **🧑/🤖 Chat History** | Persistent conversation view within the session |

---

## Setup

```bash
git clone https://github.com/Cashlin3/DocuMind--RAG-Powered-Knowledge-Assistant.git
cd DocuMind--RAG-Powered-Knowledge-Assistant
pip install -r requirements.txt
```

Add your Groq API key (free at [console.groq.com](https://console.groq.com)) to `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_key_here"
```

Then run:

```bash
streamlit run RAG.py
```

---

## Contributor

| Person | Role | Contributions |
|---|---|---|
| **Cashlin ([@Cashlin3](https://github.com/Cashlin3))** | AI Engineer | End-to-end pipeline design, LangChain-based semantic chunking, ChromaDB vector storage/retrieval, Groq LLM integration, prompt engineering for grounded generation, and Streamlit deployment |

---

## What's Next

- [ ] Cross-encoder re-ranking of retrieved chunks for higher precision
- [ ] Streaming token-by-token responses instead of blocking calls
- [ ] Multi-turn conversational memory (currently stateless per question)
- [ ] OCR fallback for scanned/image-based PDFs
