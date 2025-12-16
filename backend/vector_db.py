import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize
import uuid
import re
import numpy as np

encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="vectorstore")
collection = client.get_or_create_collection(name="study_material", metadata={"hnsw:space":"cosine"})

def _clean_text(t: str) -> str:
    if not isinstance(t, str):
        t = str(t)
    t = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def _is_code_heavy(t: str, symbol_threshold: float = 0.3) -> bool:
    if not t:
        return True
    total = max(1, len(t))
    symbols = sum(1 for c in t if not c.isalnum() and not c.isspace())
    return (symbols / total) > symbol_threshold

def process_text(text, filename, chunk_size_chars=800):
    text = _clean_text(text)
    sents = sent_tokenize(text)
    chunks = []
    cur = ""
    idx = 0
    for s in sents:
        if len(cur) + len(s) < chunk_size_chars:
            cur += " " + s
        else:
            cur = cur.strip()
            if cur and not _is_code_heavy(cur):
                chunks.append({"id": str(uuid.uuid4()), "text": cur, "source": filename, "index": idx})
                idx += 1
            cur = s
    cur = cur.strip()
    if cur and not _is_code_heavy(cur):
        chunks.append({"id": str(uuid.uuid4()), "text": cur, "source": filename, "index": idx})
    return chunks

def add_to_chroma(chunks):
    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"source": c["source"], "index": c["index"]} for c in chunks]
    if not docs:
        return
    embeddings = encoder.encode(docs).tolist()
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    rebuild_bm25()

_bm25_model=None
_bm25_texts=None
_bm25_metas=None

def rebuild_bm25():
    global _bm25_model,_bm25_texts,_bm25_metas
    all_docs = collection.get(include=["documents","metadatas"])
    texts = all_docs["documents"]
    tokenized = [t.split() for t in texts]
    if texts:
        _bm25_model = BM25Okapi(tokenized)
        _bm25_texts = texts
        _bm25_metas = all_docs["metadatas"]
    else:
        _bm25_model = None
        _bm25_texts = []
        _bm25_metas = []

def query_dense(query, n=5):
    q_emb = encoder.encode([query]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=n, include=["documents","metadatas","distances"])
    out=[]
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1/(1+dist)
        out.append({"text": doc, "source": meta["source"], "index": meta["index"], "score": float(sim)})
    return out

def query_sparse(query, n=5):
    global _bm25_model,_bm25_texts,_bm25_metas
    if _bm25_model is None:
        rebuild_bm25()
    if _bm25_model is None:
        return []
    scores = _bm25_model.get_scores(query.split())
    topk = list(np.argsort(scores)[-n:][::-1])
    out=[]
    for idx in topk:
        out.append({"text": _bm25_texts[idx], "source": _bm25_metas[idx]["source"], "index": _bm25_metas[idx]["index"], "score": float(scores[idx])})
    return out

def hybrid_retrieve(query, top_k_dense=5, top_k_sparse=5, alpha=0.7, merged_k=5):
    dense = query_dense(query, top_k_dense)
    sparse = query_sparse(query, top_k_sparse)
    combined={}
    for r in dense:
        key=f"{r['source']}:{r['index']}"
        combined[key]={"text":r["text"],"source":r["source"],"index":r["index"],"score":alpha*r["score"]}
    for r in sparse:
        key=f"{r['source']}:{r['index']}"
        if key not in combined:
            combined[key]={"text":r["text"],"source":r["source"],"index":r["index"],"score":(1-alpha)*r["score"]}
        else:
            combined[key]["score"]+= (1-alpha)*r["score"]
    merged=list(combined.values())
    merged.sort(key=lambda x:x["score"],reverse=True)

    return merged[:merged_k]

def query_vector_db(query, top_k=5):
    return hybrid_retrieve(query, merged_k=top_k)
