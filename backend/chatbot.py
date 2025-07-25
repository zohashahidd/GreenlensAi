import os
import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_db(folder_path, db_path):
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError("No PDF files found!")

    all_docs = []
    for file in pdf_files:
        loader = PyPDFLoader(os.path.join(folder_path, file))
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(all_docs)
    db = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory=db_path)
    db.persist()
    return db

# Load vector DB
db_path = "green_db"
folder_path = "reports"
db = build_vector_db(folder_path, db_path)

def ask_esg_chatbot(message):
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        request_timeout=20
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    prompt = f"""
    You are GreenLens, an assistant trained to provide ESG and sustainability insights.
    Use only the provided documents.
    - Include relevant metrics or numerical data
    - Be clear, avoid referencing document names
    - Don’t answer about companies not mentioned
    - Compare ESG performance when asked
    - Avoid email references

    Question: {message}
    """
    return qa_chain.run(prompt)
