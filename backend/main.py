# main.py (Refactored + Fixed for cheating detector & llm_meta validation)

import os
import uuid
import logging
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local modules
from extract_text import extract_text_from_file
from text_processing import process_text as chunk_text
from vector_db import add_to_chroma, hybrid_retrieve, rerank_with_cross_encoder
from graph_builder import build_graph, graph_lookup
from cheating_detector import is_cheating
from llm_client import call_llm

# -----------------------------------------------------
# App Setup
# -----------------------------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-campus-main")

app = FastAPI(title="Smart Campus Assistant – RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Pydantic Models
# -----------------------------------------------------
class ProcessResponse(BaseModel):
    message: str
    file_id: str
    filename: str
    chunks_added: int

class AskResponse(BaseModel):
    answer: str
    used_chunks: List[dict]
    sources: List[dict]
    llm_meta: Optional[dict] = None   # FIXED: allow None


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------
def _save_upload_file(upload_file: UploadFile, dest_path: str):
    with open(dest_path, "wb") as f:
        content = upload_file.file.read()
        f.write(content)

def _read_and_extract(path: str) -> str:
    if not os.path.exists(path):
        return ""
    return extract_text_from_file(path)


# -----------------------------------------------------
# 1. Upload + Process File
# -----------------------------------------------------
@app.post("/process_file", response_model=ProcessResponse)
async def process_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}_{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        _save_upload_file(file, saved_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    text = _read_and_extract(saved_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from file.")

    chunks = chunk_text(text, file.filename)

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from file.")

    add_to_chroma(chunks)
    build_graph(chunks)

    return {
        "message": "File processed successfully!",
        "file_id": file_id,
        "filename": safe_name,
        "chunks_added": len(chunks)
    }


# -----------------------------------------------------
# 1b. Process Existing File
# -----------------------------------------------------
@app.post("/process_existing", response_model=ProcessResponse)
async def process_existing_file(filename: str = Body(..., embed=True)):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    text = _read_and_extract(file_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from file.")

    chunks = chunk_text(text, filename)
    add_to_chroma(chunks)
    build_graph(chunks)

    return {
        "message": "Existing file processed successfully!",
        "file_id": filename.split("_", 1)[0] if "_" in filename else "",
        "filename": filename,
        "chunks_added": len(chunks)
    }


# -----------------------------------------------------
# 2. ASK Endpoint (RAG Pipeline)
# -----------------------------------------------------
@app.post("/ask", response_model=AskResponse)
async def ask_question(question: str = Query(...)):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1) Hybrid Retriever
    dense_sparse_results = hybrid_retrieve(
        question,
        top_k_dense=8,
        top_k_sparse=8,
        alpha=0.6,
        beta=0.4,
        merged_k=12
    )

    # 2) Reranking
    try:
        reranked = rerank_with_cross_encoder(question, dense_sparse_results, top_k=6)
    except Exception:
        reranked = dense_sparse_results[:6]

    # 3) Knowledge Graph Lookup
    graph_results = graph_lookup(question)

    # Merge context
    merged = {}
    final_context_list = []

    def k(item): return f"{item.get('source','')}_{item.get('index','')}"

    for r in reranked:
        key = k(r)
        if key not in merged:
            merged[key] = r
            final_context_list.append(r)

    for r in graph_results:
        key = k(r)
        if key not in merged:
            merged[key] = r
            final_context_list.append(r)

    # Build context text
    context_text = "\n".join([f"- {c['text']}" for c in final_context_list])

    # 4) Pre-LMM Cheating Check
    if is_cheating(question, final_context_list, ""):
        return {
            "answer": "❌ Your question requests information not found in the uploaded study materials.",
            "used_chunks": [],
            "sources": [],
            "llm_meta": {}   # FIXED: must be dict
        }

    # 5) Build LLM Prompt
    prompt = f"""
Using ONLY the provided context, answer the question.
If the question asks for applications, types, or examples, list ALL items found in context.

Context:
{context_text}

Question:
{question}

Answer concisely. Do NOT use outside knowledge.
"""

    # 6) Call LLM
    try:
        llm_resp = call_llm(prompt)
        llm_answer = llm_resp.get("answer", "").strip()
        llm_meta = llm_resp.get("meta", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # 7) Post-LMM Cheating Check
    if is_cheating(question, final_context_list, llm_answer):
        llm_answer = "❌ The requested information is NOT available in your uploaded notes."
        llm_meta = {}

    # 8) Build Response
    used_chunks = []
    for c in final_context_list:
        used_chunks.append({
            "id": c.get("id"),
            "idx": c.get("index"),
            "source": c.get("source"),
            "snippet": (c.get("text", "")[:500] + "...") if len(c.get("text","")) > 500 else c.get("text","")
        })

    return {
        "answer": llm_answer,
        "used_chunks": used_chunks,
        "sources": final_context_list,
        "llm_meta": llm_meta
    }


# -----------------------------------------------------
# 3. Summarizer
# -----------------------------------------------------
@app.post("/summarize")
async def summarize_file(filename: str = Body(..., embed=True)):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    text = _read_and_extract(path)
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted.")

    prompt = f"Summarize the following text into bullet points:\n\n{text}"

    res = call_llm(prompt)
    return {"summary": res.get("answer", "")}


# -----------------------------------------------------
# 4. Quiz Generator
# -----------------------------------------------------
@app.post("/quiz")
async def generate_quiz(filename: str = Body(..., embed=True), count: int = Body(5)):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")

    text = _read_and_extract(path)
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted.")

    prompt = f"Create {count} quiz questions (MCQ, True/False, Fill-in-the-blanks) from the text:\n\n{text}"

    res = call_llm(prompt)
    return {"quiz": res.get("answer", "")}


# -----------------------------------------------------
# 5. File List
# -----------------------------------------------------
@app.get("/files")
async def list_files():
    return {"files": os.listdir(UPLOAD_DIR)}


# -----------------------------------------------------
# 6. Delete File
# -----------------------------------------------------
@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")

    os.remove(path)
    return {"message": "File deleted."}


# -----------------------------------------------------
# Health Check
# -----------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
