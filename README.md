# ?? RAG Chat App

A Retrieval-Augmented Generation (RAG) app that lets you upload PDF or TXT files and ask questions about them using AI.

## Features
- Upload PDF and TXT files
- Ask questions about your documents
- Powered by Groq AI (free!)
- Beautiful Streamlit UI

## Setup

1. Install dependencies:
pip install streamlit pypdf2 groq chromadb

2. Get a free Groq API key at https://console.groq.com

3. Run the app:
streamlit run rag_streamlit.py

## How it works
Upload a document ? App splits it into chunks ? You ask a question ? App finds relevant chunks ? Groq AI answers using those chunks
