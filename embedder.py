from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from constant import pdf_path


def create_vector_store(pdf_path, save_path="vector_store"):
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() for page in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="multi-qa-MiniLM-L6-cos-v1",
        model_kwargs={"device": "cpu"}  # ✅ This prevents GPU/meta issues
    )
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(save_path)