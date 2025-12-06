import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="vectorstore"
))

collection = client.get_or_create_collection(
    name="study_materials",
    metadata={"hnsw:space": "cosine"}
)



def get_embedding(text):
    return embedding_model.encode(text).tolist()


def add_to_chroma(chunks):
    ids = []
    embeddings = []
    texts = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["id"])
        texts.append(chunk["text"])
        embeddings.append(get_embedding(chunk["text"]))

        metadatas.append({
            "filename": chunk["filename"],
            "index": chunk["index"]
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    client.persist()  



def query_chroma(query, n_results=3):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results
