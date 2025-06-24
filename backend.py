from groq import Groq
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from constant import groqcloud_key

client = Groq(api_key=groqcloud_key)

def load_vector_store(path):
    embeddings = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def ask_question(question, db, k=3):
    docs = db.similarity_search(question, k=k)
    context = "\n".join(doc.page_content for doc in docs)
    
    prompt = f"""
                Use the following context to answer the question:

                {context}

                Question: {question}
                """
    
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
