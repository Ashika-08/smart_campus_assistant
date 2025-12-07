import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



client = chromadb.PersistentClient(path="vectorstore")

COLLECTION_NAME = "documents"
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


bm25 = None
bm25_texts = []
bm25_chunks = []


def add_to_chroma(chunks):
    global bm25, bm25_texts, bm25_chunks

    ids = [c["id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"source": c["source"], "id": c["id"]} for c in chunks]
    embeds = encoder.encode(docs).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeds
    )

    
    bm25_texts.extend([c["text"].split() for c in chunks])
    bm25_chunks.extend(chunks)
    bm25 = BM25Okapi(bm25_texts)


def query_dense(query, n=5):
    embedding = encoder.encode([query]).tolist()

    result = collection.query(
        query_embeddings=embedding,
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    dense_results = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        dense_results.append({
            "id": meta["id"],
            "text": doc,
            "source": meta["source"],
            "score": 1 - dist
        })

    dense_results = sorted(dense_results, key=lambda x: x["score"], reverse=True)
    return dense_results


def query_sparse(query, n=5):
    if bm25 is None:
        return []

    scores = bm25.get_scores(query.split())
    sorted_ix = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for ix in sorted_ix[:n]:
        chunk = bm25_chunks[ix]
        results.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "score": float(scores[ix])
        })

    return results


def hybrid_retrieve(query, top_k_dense=5, top_k_sparse=5, alpha=0.6, beta=0.4, merged_k=8):
    dense = query_dense(query, n=top_k_dense)
    sparse = query_sparse(query, n=top_k_sparse)

    merged = {}

    
    for d in dense:
        merged[d["id"]] = d["score"] * alpha

    
    for s in sparse:
        if s["id"] in merged:
            merged[s["id"]] += s["score"] * beta
        else:
            merged[s["id"]] = s["score"] * beta

    
    final = []
    for chunk_id, score in merged.items():
        doc = collection.get(ids=[chunk_id])
        final.append({
            "id": chunk_id,
            "text": doc["documents"][0],
            "source": doc["metadatas"][0]["source"],
            "score": score
        })

    final = sorted(final, key=lambda x: x["score"], reverse=True)
    return final[:merged_k]


def rerank_with_cross_encoder(query, results, top_k=5):
    if not results:
        return []

    pairs = [(query, r["text"]) for r in results]
    scores = cross_encoder.predict(pairs)

    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])

    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]


def get_chunk_by_id(chunk_id):
    doc = collection.get(ids=[chunk_id])
    if not doc["documents"]:
        return None
    return {
        "id": chunk_id,
        "text": doc["documents"][0],
        "source": doc["metadatas"][0]["source"]
    }
