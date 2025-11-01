import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Embeddings
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("📄 Conversational RAG with PDF & Chat History")
st.write("Upload a PDF and chat with its content!")

# Groq key
api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-20b")

    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # PROMPTS
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", 
                "Rewrite user query to be standalone using chat history if needed. "
                "Do NOT answer, only rewrite."
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        system_prompt = (
            "You are a helpful AI assistant. Use the context below to answer.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_retriever, answer_chain)

        # Session logic
        def get_session_history(session_id) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # user input
        user_input = st.text_input("Ask a question about the PDF:")

        if user_input:
            session_history = get_session_history(session_id)

            response = conversational_rag.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.success(f"**Assistant:** {response['answer']}")
            st.write("🧠 **Chat History:**", session_history.messages)

else:
    st.warning("⚠️ Please enter your Groq API key to continue.")
