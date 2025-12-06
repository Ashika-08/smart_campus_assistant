import os
from fastapi import FastAPI, UploadFile, File
import shutil
from extract_text import extract_text
import ollama
from text_processing import process_text     
from vector_db import add_to_chroma ,query_chroma




app = FastAPI()
UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    extracted = extract_text(file_path)

    return {
        "filename": file.filename,
        "extracted_preview": extracted[:300],
        "full_text_length": len(extracted),
        "status": "success"
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
        "status": "stored_in_vector_db"
    }


@app.post("/ask")
async def ask_question(question: str):
    results = query_chroma(question)

    retrieved_chunks = results["documents"][0]
    combined_context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an AI tutor. Answer ONLY using this context:

{combined_context}

Question:
{question}

Give a clear and correct answer.
"""

    response = ollama.chat(
        model="llama3:latest",
        messages=[
            {"role": "system", "content": "Use only provided context. Do not guess."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "chunks_used": retrieved_chunks
    }
@app.post("/summarize")
async def summarize_text(filename: str):
    # Load extracted text
    file_path = f"uploads/{filename}"
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    text = extract_text(file_path)

    # Build summarization prompt
    prompt = f"""
Summarize the following study material into clear, concise points.
Keep only important concepts.

Text:
{text}
"""

    response = ollama.chat(
        model="llama3:latest",
        messages=[
            {"role": "system", "content": "You summarize educational content clearly."},
            {"role": "user", "content": prompt}
        ],
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
Create {num_questions} quiz questions from the following study content.
Include a mix of:
- Multiple-choice questions
- True/False
- Short answer

Make questions clear and relevant. Provide answers separately.

Text:
{text}
"""

    response = ollama.chat(
        model="llama3:latest",
        messages=[
            {"role": "system", "content": "You create quizzes for students."},
            {"role": "user", "content": prompt}
        ],
    )

    quiz = response["message"]["content"]

    return {"quiz": quiz}
