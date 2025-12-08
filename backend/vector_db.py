# vector_db.py (patched version)

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

import uuid

# -------------------------------------------
# 1) Load Models
# -------------------------------------------
encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------------------
# 2) Persistent Chroma Client
# -------------------------------------------
client = chromadb.PersistentClient(path="vectorstore")

collection = client.get_or_create_collection(
    name="study_material",
    metadata={"hnsw:space": "cosine"}  # cosine distance
)

# -------------------------------------------
# 3) BM25 Globals (Incremental!)
# -------------------------------------------
bm25_model = None
bm25_texts = []
bm25_metas = []
bm25_tokenized = []

def _rebuild_bm25_index():
    """
    Rebuild BM25 index when needed.
    Called only when new docs are added (not on every query).
    """
    global bm25_model
    if not bm25_texts:
        bm25_model = None
        return
    bm25_model = BM25Okapi(bm25_tokenized)

# -------------------------------------------
# 4) Add Chunks (vector + sparse storage)
# -------------------------------------------
def add_to_chroma(chunks):
    global bm25_texts, bm25_metas, bm25_tokenized, bm25_model

    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"source": c["source"], "index": c["index"], "id": c["id"]} for c in chunks]

    # Embeddings
    embeddings = encoder.encode(docs).tolist()

    # Add to Chroma
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings
    )

    # Incrementally update BM25
    for text, meta in zip(docs, metas):
        tokens = text.split()
        bm25_texts.append(text)
        bm25_metas.append(meta)
        bm25_tokenized.append(tokens)

    # Rebuild once
    _rebuild_bm25_index()


# -------------------------------------------
# 5) Dense Retriever
# -------------------------------------------
def query_dense(query, n=5):
    q_emb = encoder.encode([query]).tolist()

    res = collection.query(
        query_embeddings=q_emb,
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    dense_results = []
    for doc, meta, dist in zip(
        res["documents"][0], 
        res["metadatas"][0], 
        res["distances"][0]
    ):
        sim = 1 - dist  # FIXED cosine similarity

        dense_results.append({
            "id": meta["id"],
            "text": doc,
            "source": meta["source"],
            "index": meta["index"],
            "score": float(sim)
        })

    return dense_results


# -------------------------------------------
# 6) Sparse Retriever (BM25)
# -------------------------------------------
def query_sparse(query, n=5):
    global bm25_model

    if bm25_model is None or not bm25_texts:
        return []

    scores = bm25_model.get_scores(query.split())
    top_idx = scores.argsort()[-n:][::-1]

    sparse_results = []
    for idx in top_idx:
        sparse_results.append({
            "id": bm25_metas[idx]["id"],
            "text": bm25_texts[idx],
            "source": bm25_metas[idx]["source"],
            "index": bm25_metas[idx]["index"],
            "score": float(scores[idx])
        })

    return sparse_results


# -------------------------------------------
# 7) Hybrid
# -------------------------------------------
def hybrid_retrieve(query, top_k_dense=5, top_k_sparse=5, alpha=0.6, beta=0.4, merged_k=6):
    dense = query_dense(query, top_k_dense)
    sparse = query_sparse(query, top_k_sparse)

    combined = {}

    # Combine dense
    for r in dense:
        combined[r["id"]] = {
            **r,
            "score": alpha * r["score"]
        }

    # Combine sparse
    for r in sparse:
        if r["id"] not in combined:
            combined[r["id"]] = {
                **r,
                "score": beta * r["score"]
            }
        else:
            combined[r["id"]]["score"] += beta * r["score"]

    # Sort
    merged = list(combined.values())
    merged.sort(key=lambda x: x["score"], reverse=True)

    return merged[:merged_k]


# -------------------------------------------
# 8) Cross-Encoder Reranker
# -------------------------------------------
def rerank_with_cross_encoder(query, results, top_k=3):
    pairs = [(query, r["text"]) for r in results]
    scores = cross_encoder.predict(pairs)

    reranked = []
    for r, score in zip(results, scores):
        new_r = r.copy()
        new_r["score"] = float(score)
        reranked.append(new_r)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]
