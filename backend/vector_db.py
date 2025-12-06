import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


chroma_client = chromadb.PersistentClient(path="vectorstore")


collection = chroma_client.get_or_create_collection(
    name="notes",                     
    metadata={"hnsw:space": "cosine"} 
)


def add_to_chroma(chunks):
    """
    chunks = [
        {
            "id": "unique-uuid",
            "text": "chunk text",
            "source": "filename",
        },
        ...
    ]
    """

    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]

    
    embeddings = model.encode(texts).tolist()

    
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return True



def query_chroma(question, n=5):
    """
    Returns top-n matching chunks for a user question.
    """

    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n
    )

    return results
