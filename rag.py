import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader 
from sentence_transformers import SentenceTransformer


DOCS_DIR = Path("docs")
INDEX_DIR = Path("index")
Embed_model = "all-MiniLM-L6-V2"
Chunk_Size = 800
Chunk_Overlap = 150
Top_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the provided documents. "
    "Use ONLY the context given. "
    "Write your answer as one clear paragraph of 4 to 6 complete sentences, in plain prose. "
    "Do not use bullet points, lists, or headings. "
    "Combine details from all relevant passages into a single flowing explanation, "
    "and cite the passages you used as [1], [2], etc. "
    "If the context only partly answers the question, explain what it does show. "
    "If it contains nothing relevant, reply only with: "
    "'I could not find that in the provided documents.'"
)

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(Embed_model)
    return _embedder

def load_documents (folder=DOCS_DIR):
    pages = []
    for path in sorted(Path(folder).glob("*")):
        suffix = path.suffix.lower()
        if suffix ==".pdf":
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"source": path.name, "page": i, "text": text})
        elif suffix in {".txt", ".md"}:
            pages.append(
                {"source": path.name, "page": 1, "text": path.read_text(encoding="utf-8" )}
            )
    return pages 

def chunk_text(text, size = Chunk_Size, overlap=Chunk_Overlap):
    text = " " .join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start+size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks 


def embed(texts):
    vectors = get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype="float32")

def build_index (folder=DOCS_DIR):
    chunks =[]
    for page in load_documents(folder):
        for piece in chunk_text(page["text"]):
            chunks.append({"source": page["source"], "page": page["page"], "text":piece})
    
    if not chunks:
        raise ValueError(f"No .pdf/.txt/.md documents found in '{folder}'.")
    
    
    vectors = embed ([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index,str(INDEX_DIR/ "faiss.index"))
    (INDEX_DIR/ "chunks.json").write_text(json.dumps(chunks),encoding="utf-8")
    return len(chunks)


def load_index():
    index = faiss.read_index(str(INDEX_DIR/ "faiss.index"))
    chunks = json. loads((INDEX_DIR/"chunks.json").read_text(encoding="utf-8"))
    return index, chunks 

Rerank_Model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None 


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        
        _reranker = CrossEncoder(Rerank_Model)
    return _reranker
 
 
 
def retrieve(query, index, chunks, k=Top_K, rerank=False, candidates=20) :
    n = candidates if rerank else k
    scores, ids = index.search(embed([query]), n)
    results = [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i !=-1
    ]
    
    if rerank and results:
        new_scores = get_reranker().predict([(query, r["text"]) for r in results])
        for r, s in zip(results, new_scores):
            r["scores"] = float(s)
        results.sort(key=lambda r: r["scores"], reverse=True)
        results = results[:k]
    return results


def generate(prompt):
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    
    if provider == "anthropic":
        from anthropic import Anthropic 
        
        client = Anthropic()
        response = client.messages.create(
            model = os.getenv("Anthropic_model", "claude-sonnet-5"),
            max_tokens=700,
            system = SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    if provider == "ollama":
        import requests
        
        response = requests.post(
         "http://localhost:11434/api/chat",
         json ={
             "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
             "stream": False,
             "messages": [
                 {"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt},
             ],
         },
          timeout=120,
        )
        return response.json()["message"]["content"]
    raise ValueError(f"UNknown LLM_PROVIDER: {provider}")


def answer (question, index, chunks, k = Top_K, rerank= False):
    hits = retrieve(question, index, chunks, k, rerank=rerank)
    context ="\n\n".join(
        f"[{n}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for n, h in enumerate(hits, start=1)
    )
    prompt = f"Content:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return generate(prompt), hits


if __name__ =="__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        print(f"indexed {build_index()} chunks.")
    
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        idx, chs =load_index()
        reply, source = answer(" ". join(sys.argv[2:]), idx,chs)
        print(reply, "\n\nSources:")
        for s in source:
            print(f"-  {s['source']} (page {s['page']}, score {s['score']:.2f})")
    else:
        print('Usage:\n  python rag.py ingest\n  python rag.py ask "your question"')
        
            