# Chat with Your Documents (RAG)

Ask questions about your own PDFs and get answers with source citations.
Built from scratch without LangChain.

## How it works
Documents → chunking → embeddings (sentence-transformers) → FAISS index → top-k retrieval
→ optional cross-encoder re-ranking → LLM (Ollama or Claude API) → answer with sources

## Tech stack
Python, sentence-transformers, FAISS, pypdf, Streamlit, Ollama / Claude API

python -m venv venv
venv/bin/activate       
pip install -r requirements.txt
```
Install [Ollama](https://ollama.com) and run `ollama pull llama3.2

python rag.py ingest           
python rag.py ask "your question"
streamlit run app.py
python eval.py                 


## Results
Retrieval hit-rate on a 15-question test set: @1 = 40%, @3 = 60%, @5 = 67% (vector search only).
