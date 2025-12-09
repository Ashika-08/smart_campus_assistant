# vector_db.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize
import uuid


encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


client = chromadb.PersistentClient(path="vectorstore")

collection = client.get_or_create_collection(
    name="study_material",
    metadata={"hnsw:space": "cosine"}
)



def process_text(text, filename):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    chunk_index = 0

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:
            current_chunk += " " + sentence
        else:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": current_chunk.strip(),
                "source": filename,
                "index": chunk_index
            })
            chunk_index += 1
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": current_chunk.strip(),
            "source": filename,
            "index": chunk_index
        })

    return chunks



def add_to_chroma(chunks):
    global bm25_model, bm25_texts, bm25_metas

    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"source": c["source"], "index": c["index"]} for c in chunks]

    embeddings = encoder.encode(docs).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings
    )

    bm25_model = None
    bm25_texts = None
    bm25_metas = None



def get_chunk_by_id(chunk_id):
    res = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not res["documents"]:
        return None
    return {
        "text": res["documents"][0],
        "source": res["metadatas"][0]["source"],
        "index": res["metadatas"][0]["index"],
    }



def query_dense(query, n=5):
    q_emb = encoder.encode([query]).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    dense_results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 / (1 + dist)  # convert distance → similarity
        dense_results.append({
            "text": doc,
            "source": meta["source"],
            "index": meta["index"],
            "score": float(sim)
        })

    return dense_results



def build_bm25_index():
    all_docs = collection.get(include=["documents", "metadatas"])
    texts = all_docs["documents"]
    tokenized = [t.split() for t in texts]
    return BM25Okapi(tokenized), texts, all_docs["metadatas"]


bm25_model = None
bm25_texts = None
bm25_metas = None


def ensure_bm25_index():
    global bm25_model, bm25_texts, bm25_metas
    if bm25_model is None:
        bm25_model, bm25_texts, bm25_metas = build_bm25_index()


def query_sparse(query, n=5):
    ensure_bm25_index()
    scores = bm25_model.get_scores(query.split())
    top_idx = scores.argsort()[-n:][::-1]

    sparse_results = []
    for idx in top_idx:
        sparse_results.append({
            "text": bm25_texts[idx],
            "source": bm25_metas[idx]["source"],
            "index": bm25_metas[idx]["index"],
            "score": float(scores[idx])
        })

    return sparse_results



def hybrid_retrieve(query, top_k_dense=5, top_k_sparse=5, alpha=0.6, beta=0.4, merged_k=5):
    dense = query_dense(query, top_k_dense)
    sparse = query_sparse(query, top_k_sparse)

    combined = {}

    for r in dense:
        key = f"{r['source']}:{r['index']}"
        combined[key] = {
            "text": r["text"],
            "source": r["source"],
            "index": r["index"],
            "score": alpha * r["score"],
        }

    for r in sparse:
        key = f"{r['source']}:{r['index']}"
        if key not in combined:
            combined[key] = {
                "text": r["text"],
                "source": r["source"],
                "index": r["index"],
                "score": beta * r["score"],
            }
        else:
            combined[key]["score"] += beta * r["score"]

    merged = list(combined.values())
    merged.sort(key=lambda x: x["score"], reverse=True)

    return merged[:merged_k]



def rerank_with_cross_encoder(query, results, top_k=3):
    pairs = [(query, r["text"]) for r in results]
    scores = cross_encoder.predict(pairs)

    reranked = []
    for r, s in zip(results, scores):
        nr = r.copy()
        nr["score"] = float(s)
        reranked.append(nr)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


# -----------------------------
# REQUIRED BY KNOWLEDGE GRAPH MODULE
# → Simple wrapper for hybrid vector search
# -----------------------------
def query_vector_db(query: str, top_k: int = 5):
    """
    Used by Knowledge Graph module for vector retrieval.
    """
    results = query_dense(query, n=top_k)
    output = []
    for r in results:
        output.append({
            "text": r["text"],
            "score": r["score"],
            "metadata": {"source": r["source"], "index": r["index"]}
        })
    return output
