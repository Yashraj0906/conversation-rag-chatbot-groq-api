# Conversation RAG Chatbot (Groq API)

A conversational Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, and the Groq API.  
Upload PDFs, ask questions, and get conversational answers with context and history support.

## 🚀 Features

- Upload one or more PDF documents and automatically split them into searchable chunks  
- Embed the text using Hugging Face embeddings (`all-MiniLM-L6-v2`)  
- Store embeddings in a vector store (Chroma) for efficient retrieval  
- Use the Groq API as the LLM backend for generation  
- History-aware retriever: follow-up questions are rewritten to be standalone using chat history  
- Persistent chat history per session (allows conversational context)  
- Streamlit UI: upload files, ask questions, view chat history  
- Clean code structure and environment setup for quick deployment

## 🧩 Stack / Tools

| Component | Description |
|-----------|-------------|
| Python | 3.10+ recommended |
| Streamlit | Web UI for interaction |
| LangChain Classic | Retrieval & chain orchestration |
| HuggingFaceEmbeddings | Text → vector embedding |
| Chroma | Local vector store for retrieval |
| Groq API (`ChatGroq`) | LLM backend for answer generation |
| PyPDFLoader | Load PDF documents into LangChain documents |

## 📂 Project Structure

├── .gitignore

├── requirements.txt

├── app1.py # Main Streamlit app (rename as needed)

├── app.py # (Optional) earlier version / archive 
└── … # other files e.g. .venv, temp files


> **Note:** Files like `.env`, uploaded PDFs, and temporary files are included in `.gitignore` for safety.

## 🛠 Setup & Run

1. Clone the repository  
   ```bash
   git clone https://github.com/Yashraj0906/conversation-rag-chatbot-groq-api.git
   cd conversation-rag-chatbot-groq-api
   
Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate     # on Windows use: .venv\Scripts\activate

Create a .env file at the root with your secret keys:

GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here

Run the Streamlit app

streamlit run app1.py
