from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from vector_db import (
    process_text,
    add_to_chroma,
    hybrid_retrieve,
    rerank_with_cross_encoder,
)
import ollama
import json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def detect_hallucination(answer: str, context_chunks: list) -> bool:
    """
    Returns True if the LLM appears to add new information
    that does NOT appear in the retrieved chunks.
    """

    
    context_text = " ".join([c["text"].lower() for c in context_chunks])
    ans = answer.lower()

   
    important_words = [w for w in ans.split() if len(w) > 4]
    missing = [w for w in important_words if w not in context_text]

    if len(missing) > 6:  
        return True

    
    patterns = ["4x4", "binary tree", "linked list", "database", "sorting"]
    for p in patterns:
        if p in ans and p not in context_text:
            return True

   
    if ("code" in ans or "example" in ans) and "code" not in context_text:
        return True

    return False



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"filename": file.filename, "status": "uploaded"}



@app.post("/process_file")
async def process_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

   
    from text_extract import extract_text
    text = extract_text(file_path)

    chunks = process_text(text, filename)
    add_to_chroma(chunks)

    return {"status": "processed", "chunks": len(chunks)}


@app.post("/ask")
async def ask_question(question: str):
    # --- STEP 1: Hybrid retrieve ---
    hybrid_results = hybrid_retrieve(
        question,
        top_k_dense=5,
        top_k_sparse=5,
        merged_k=8
    )

   
    reranked = rerank_with_cross_encoder(question, hybrid_results, top_k=5)

    
    context_text = "\n\n".join([f"[Chunk {i}] {r['text']}" for i, r in enumerate(reranked)])

    
    prompt = f"""
You are a teaching assistant. Answer ONLY using the information in the provided document chunks.
If the answer is not present in the chunks, say:

"I could not find the answer in the uploaded document."

QUESTION:
{question}

DOCUMENT CHUNKS:
{context_text}

ANSWER:
"""

    response = ollama.generate(
        model="llama3",
        prompt=prompt
    )
    answer_text = response["response"].strip()

    
    hallucinated = detect_hallucination(answer_text, reranked)

    if hallucinated:
        answer_text = (
            "The answer is **NOT present in the uploaded document**. "
            "So I cannot answer confidently based on provided material."
        )

   
    sources = [
        {"source": r["source"], "index": r["index"], "score": r["score"]}
        for r in reranked
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "used_chunks": [{"idx": i, "source": r["source"]} for i, r in enumerate(reranked)],
        "confidence": reranked[0]["score"],
        "llm_meta": {"provider": "ollama_cli"},
    }
