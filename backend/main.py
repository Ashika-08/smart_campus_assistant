# main.py
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List, Dict, Any

from extract_text import extract_text
from vector_db import (
    process_text,
    add_to_chroma,
    hybrid_retrieve,
    rerank_with_cross_encoder,
)
from llm_client import call_llm

app = FastAPI()

# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -----------------------
# Hallucination detector (keeps simple)
# -----------------------
def detect_hallucination(answer: str, chunks: List[Dict[str, Any]]) -> bool:
    context_text = " ".join([c["text"].lower() for c in chunks])
    ans = answer.lower()
    important_words = [w for w in ans.split() if len(w) > 4]
    missing = [w for w in important_words if w not in context_text]
    if len(missing) > 10:
        return True
    if ("example" in ans or "code" in ans) and "code" not in context_text:
        return True
    return False


# -----------------------
# Basic endpoints: upload / process_file / ask
# -----------------------
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
    text = extract_text(file_path)
    chunks = process_text(text, filename)
    add_to_chroma(chunks)
    return {"status": "processed", "chunks": len(chunks)}


@app.post("/ask")
async def ask_question(question: str):
    hybrid_results = hybrid_retrieve(question, top_k_dense=5, top_k_sparse=5, merged_k=8)
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
    llm_response = call_llm(prompt, model="llama3")
    answer_text = llm_response["answer"].strip()
    hallucinated = detect_hallucination(answer_text, reranked)
    if hallucinated:
        answer_text = "The answer is NOT present in the uploaded document, so I cannot answer confidently."
    sources = [{"source": r["source"], "index": r["index"], "score": r["score"]} for r in reranked]
    return {
        "answer": answer_text,
        "sources": sources,
        "used_chunks": [{"idx": i, "source": r["source"]} for i, r in enumerate(reranked)],
        "confidence": reranked[0]["score"] if reranked else 0,
        "llm_meta": llm_response.get("meta", {}),
    }


@app.post("/summarize")
async def summarize_file(filename: str, mode: str = "short", max_chunks: int = 30, batch_size: int = 6):
    """
    Summarize a file using map-reduce summarization.
    Hallucination detection is DISABLED because summaries naturally paraphrase.
    """

    results = hybrid_retrieve(filename, top_k_dense=50, top_k_sparse=50, merged_k=200)
    file_chunks = [r for r in results if r.get("source") == filename]

    if not file_chunks:
        return {"error": "No chunks found for file. Did you run /process_file?"}

    top_chunks = file_chunks[:min(len(file_chunks), max_chunks)]

    # ---------------- MAP STEP ----------------
    chunk_summaries = []
    for i in range(0, len(top_chunks), batch_size):
        batch = top_chunks[i:i+batch_size]
        context_text = "\n\n".join([f"[Chunk {c['index']}] {c['text']}" for c in batch])

        map_prompt = f"""
You are an expert summarizer. Produce a concise {mode} summary of the following chunks.
Do NOT add facts not present.

CHUNKS:
{context_text}

Return ONLY the summary.
"""

        map_out = call_llm(map_prompt, model="llama3")
        chunk_summaries.append(map_out["answer"].strip())

    # ---------------- REDUCE STEP ----------------
    combined = "\n\n".join(chunk_summaries)

    reduce_prompt = f"""
Combine the following partial summaries into ONE unified {mode} summary.
Remove duplicates. Keep it factual. No extra information.

SUMMARIES:
{combined}

Return ONLY the final summary.
"""

    final_out = call_llm(reduce_prompt, model="llama3")
    final_summary = final_out["answer"].strip()

    # NOTE: Hallucination check disabled for summarization.
    return {
        "summary": final_summary,
        "num_chunks_used": len(top_chunks),
        "chunk_summaries": chunk_summaries,
        "llm_meta": final_out.get("meta", {}),
    }


@app.post("/generate_quiz")
async def generate_quiz(
    filename: str,
    num_questions: int = 10,
    difficulty: str = "medium",
    max_chunks: int = 20
):
    """
    Fully polished quiz generator:
    - MCQ, True/False, Fill-in-the-blank
    - Grounded answers only (no hallucination)
    - JSON-safe output with auto-repair
    - Boolean answers converted to strings
    """

    import json, re

    # ---------------- 1) RETRIEVE CHUNKS ----------------
    results = hybrid_retrieve(filename, top_k_dense=50, top_k_sparse=50, merged_k=100)
    file_chunks = [r for r in results if r["source"] == filename]

    if not file_chunks:
        return {"error": "No chunks found. Did you run /process_file first?"}

    top_chunks = file_chunks[:min(max_chunks, len(file_chunks))]
    context_text = "\n\n".join([f"[Chunk {c['index']}] {c['text']}" for c in top_chunks])

    # ---------------- 2) QUIZ PROMPT ----------------
    quiz_prompt = f"""
You are an expert educational content creator.
Generate EXACTLY {num_questions} quiz questions using ONLY the content below.

CONTENT:
{context_text}

QUESTION TYPES REQUIRED:
- MCQ (1 correct + 3 plausible distractors)
- True/False
- Fill-in-the-blank

FOR EACH QUESTION, OUTPUT A JSON OBJECT EXACTLY LIKE:
{{
  "id": "<unique id>",
  "type": "mcq" | "tf" | "blank",
  "question": "<question>",
  "options": ["A","B","C","D"],   # Only for MCQ
  "answer": "<correct answer>",
  "explanation": "Source: Chunk X",
  "source": {{
    "file": "{filename}",
    "chunk_index": <chunk index>
  }}
}}

RULES:
1. Use ONLY the context.
2. Distractors MUST be grounded.
3. Explanation MUST reference correct chunk index.
4. Output MUST be a JSON ARRAY ONLY.
5. DO NOT add any text outside JSON.
"""

    # ---------------- 3) CALL LLM ----------------
    llm_out = call_llm(quiz_prompt, model="llama3")
    raw_output = llm_out.get("answer", "")

    parsed = None

    # ---------------- 4) TRY DIRECT JSON PARSE ----------------
    try:
        parsed = json.loads(raw_output)
    except:
        parsed = None

    # ---------------- 5) TRY SLICE JSON ----------------
    if parsed is None:
        try:
            start = raw_output.find("[")
            end = raw_output.rfind("]") + 1
            sliced = raw_output[start:end]
            parsed = json.loads(sliced)
        except:
            parsed = None

    # ---------------- 6) ASK LLM TO FIX JSON ----------------
    if parsed is None:
        try:
            fix_prompt = f"""
The following is invalid JSON. Convert it into a VALID JSON array ONLY.

OUTPUT:
{raw_output}
"""
            fixed = call_llm(fix_prompt, model="llama3")
            reformatted = fixed.get("answer", "")
            parsed = json.loads(reformatted)
        except:
            return {
                "error": "JSON_PARSE_FAILED",
                "raw": raw_output,
                "fixed_attempt": reformatted if 'reformatted' in locals() else None,
                "meta": llm_out.get("meta", {})
            }

    # ---------------- 7) GROUNDING CHECK ----------------
    context_lower = " ".join([c["text"].lower() for c in top_chunks])
    final_questions = []

    for q in parsed:
        if not isinstance(q, dict):
            continue

        question = q.get("question", "").strip()
        answer = q.get("answer", "")
        qtype = q.get("type", "").lower()

        if not question:
            continue

        # Convert answer to lowercase string always
        answer_str = str(answer).strip().lower()
        if not answer_str:
            continue

        tokens = re.findall(r"[a-zA-Z0-9]{4,}", answer_str)
        grounded = any(t in context_lower for t in tokens)

        # MCQ + blank → must be grounded
        if qtype in ("mcq", "blank") and not grounded:
            continue

        # True/False → normalize and validate
        if qtype == "tf":
            if answer_str not in ("true", "false"):
                continue

        # Ensure proper source structure
        if "source" not in q or not isinstance(q["source"], dict):
            q["source"] = {
                "file": filename,
                "chunk_index": top_chunks[0]["index"]
            }

        final_questions.append(q)

        if len(final_questions) >= num_questions:
            break

    # ---------------- 8) NORMALIZE ANSWERS FOR FRONTEND ----------------
    for q in final_questions:
        q["answer"] = str(q.get("answer", ""))  # Convert bool → string

    # ---------------- 9) RETURN ----------------
    return {
        "quiz": final_questions,
        "requested": num_questions,
        "generated": len(final_questions),
        "used_chunks": [c["index"] for c in top_chunks],
        "llm_meta": llm_out.get("meta", {})
    }
