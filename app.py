import streamlit as st
from backend import ask_question, load_vector_store
from embedder import create_vector_store
import os

st.title("📄 Chat with your Document (Groq + HuggingFace)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    # Save to temp
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Re-embed and store
    with st.spinner("Processing document..."):
        create_vector_store("uploaded.pdf")
        st.session_state.db = load_vector_store("vector_store")

# Load vector store only once
if 'db' not in st.session_state and not uploaded_file:
    with st.spinner("Loading default vector store..."):
        st.session_state.db = load_vector_store("vector_store")

# Ask a question
query = st.text_input("Ask a question:")
if query and st.session_state.get("db"):
    with st.spinner("Thinking..."):
        answer = ask_question(query, st.session_state.db)
        st.markdown(f"**Answer:** {answer}")
