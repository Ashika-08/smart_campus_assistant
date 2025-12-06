from fastapi import FastAPI, UploadFile, File
import shutil, os
from extract_text import extract_text
from extract_text import extract_text        
from text_processing import process_text     
from vector_db import add_to_chroma 

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

