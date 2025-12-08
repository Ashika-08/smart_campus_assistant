# main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List
from extract_text import extract_text
from text_processing import process_text
from vector_db import add_to_chroma, hybrid_retrieve, rerank_with_cross_encoder, get_chunk_by_id


try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Smart RAG Backend (v1.5)")


def call_llm(prompt: str, model: str = "llama3:latest", max_tokens: int = 512):
    """
    Calls local Ollama. If ollama not available, raise an error so user can swap in OpenAI or another client.
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("ollama package not available. Install ollama python package OR replace call_llm with your preferred LLM client.")
    
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(file_path)
    preview = text[:400]
    return {"filename": file.filename, "preview": preview, "full_text_length": len(text)}


async def process_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    
    extracted = extract_text(file_path)

   
    chunks = process_text(extracted, filename=file.filename)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from document.")

   
    add_to_chroma(chunks)

    return {"filename": file.filename, "chunks_stored": len(chunks), "status": "stored_in_vector_db"}


@app.get("/ask")
async def ask_question(question: str = Query(..., min_length=1),
                       dense_k: int = 5,
                       sparse_k: int = 10,
                       merged_k: int = 7,
                       rerank_k: int = 5):
    """
    Steps:
     1) Hybrid retrieve (dense + sparse)
     2) Rerank with cross-encoder
     3) Build context from top reranked chunks (with citations)
     4) Call LLM to answer using only the provided context
    """
    merged = hybrid_retrieve(query=question,
                             top_k_dense=dense_k,
                             top_k_sparse=sparse_k,
                             alpha=0.6,
                             beta=0.4,
                             merged_k=merged_k)

    if not merged:
        return {"answer": "", "used_chunks": [], "confidence": 0.0, "note": "No candidates found."}

    
    reranked = rerank_with_cross_encoder(question, merged, top_k=rerank_k)

    
    top_texts = []
    used_chunks = []
    for item in reranked:
        top_texts.append(f"[Source: {item['source']} | idx: {item.get('index', 'n/a')}]\n{item['text']}")
        used_chunks.append({"source": item["source"], "index": item.get("index", None), "score": item.get("score", None)})

    combined_context = "\n\n---\n\n".join(top_texts)

    
    prompt = f"""
You are an expert tutor. Answer the question ONLY using the provided CONTEXT. Do NOT hallucinate.
If the context does not contain the answer, state: "The document does not contain the answer."

CONTEXT:
{combined_context}

QUESTION:
{question}

Answer concisely and include short numbered citations in the form [source|index] where relevant.
"""

   
    try:
        answer = call_llm(prompt)
    except Exception as e:
       
        return JSONResponse(status_code=500, content={"error": "LLM call failed", "detail": str(e), "candidates": used_chunks})

    
    scores = [c.get("score", 0.0) for c in reranked if isinstance(c.get("score", None), (int, float))]
    confidence = float(sum(scores) / len(scores)) if scores else 0.0

    return {"answer": answer, "used_chunks": used_chunks, "confidence": confidence}


@app.get("/files")
async def list_files():
    files = os.listdir(UPLOAD_DIR)
    return {"files": files}

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"deleted": filename}
    else:
        raise HTTPException(status_code=404, detail="File not found")
