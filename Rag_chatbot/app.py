
import streamlit as st
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot (Multi-file + PDF)")

# ---------------- API KEY ----------------
os.environ["OPENAI_API_KEY"] = "API_KEY"

# ---------------- LOAD DOCUMENTS ----------------
def load_documents():

    docs = []
    data_path = Path(__file__).parent / "data"
    
    for file_path in data_path.iterdir():
        if file_path.is_file():
            if file_path.suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs.extend(loader.load())

            elif file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs.extend(loader.load())

    return docs

# ---------------- BUILD RAG PIPELINE ----------------
@st.cache_resource
def load_rag():
    documents = load_documents()
    st.write(f"Loaded {len(documents)} documents")

    data_path = Path(__file__).parent / "data"
    st.write("📂 Files being processed:")
    for file_path in data_path.iterdir():
        st.write(file_path.name)

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    st.write(f"Total chunks: {len(docs)}")

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3})
    )

   
    return qa

qa = load_rag()

# ---------------- CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask something from your documents...")

if user_input:
    # user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # AI response
    response = qa.run(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)