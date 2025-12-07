from fastapi import FastAPI, UploadFile, File
import os, shutil

from extract_text import extract_text
from text_processing import process_text
from vector_db import (
    add_to_chroma,
    hybrid_retrieve,
    rerank_with_cross_encoder,
    get_chunk_by_id
)

import ollama

app = FastAPI()

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


DENSE_K = 10
SPARSE_K = 10
MERGED_K = 10
RERANK_TOPK = 5



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(file_path)

    return {
        "filename": file.filename,
        "preview": text[:300],
        "status": "uploaded",
    }



@app.post("/process_file")
async def process_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted = extract_text(file_path)
    chunks = process_text(extracted, filename=file.filename)

    add_to_chroma(chunks)

    return {
        "filename": file.filename,
        "chunks_stored": len(chunks),
        "status": "indexed",
    }



@app.post("/ask")
async def ask_question(question: str):
    # 1️⃣ HYBRID search (dense + sparse)
    merged = hybrid_retrieve(
        query=question,
        top_k_dense=DENSE_K,
        top_k_sparse=SPARSE_K,
        alpha=0.6,
        beta=0.4,
        merged_k=MERGED_K
    )

    if len(merged) == 0:
        return {"answer": "No relevant information found.", "sources": []}

   
    reranked = rerank_with_cross_encoder(
        question,
        merged,
        top_k=RERANK_TOPK       
    )

   
    final_context = "\n\n".join([f"[Chunk {i+1}] {item['text']}" for i, item in enumerate(reranked)])

    prompt = f"""
You are a helpful study assistant. Answer ONLY using the context below.
If the answer is not in the context, say "The document does not contain this information."

Context:
{final_context}

Question:
{question}

Provide a clear and short explanation.
Also include citations like: [Chunk 1], [Chunk 2].
"""

    response = ollama.chat(
        model="llama3:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources": reranked
    }



@app.post("/summarize")
async def summarize_text(filename: str):
    file_path = f"uploads/{filename}"
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    text = extract_text(file_path)

    prompt = f"""
Summarize the following content into clear, simple bullet points.

Text:
{text}
"""

    response = ollama.chat(
        model="llama3:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    summary = response["message"]["content"]
    return {"summary": summary}



@app.post("/quiz")
async def generate_quiz(filename: str, num_questions: int = 5):
    file_path = f"uploads/{filename}"
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    text = extract_text(file_path)

    prompt = f"""
Create {num_questions} quiz questions from this text.
Make a mix of MCQ, True/False, and Short Answer.
Give answers separately.

Text:
{text}
"""

    response = ollama.chat(
        model="llama3:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"quiz": response["message"]["content"]}



@app.get("/files")
async def list_files():
    return {"files": os.listdir(UPLOAD_DIR)}



@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    os.remove(file_path)
    return {"status": "deleted", "filename": filename}
