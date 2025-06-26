### ✅ Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Create your `constant` file with API keys
3. Run embedding once:
```bash
python -c "from embedder import create_vector_store; create_vector_store('your.pdf')"
```
4. Launch UI:
```bash
streamlit run app.py
