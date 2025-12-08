# text_processing.py
import uuid
from nltk.tokenize import sent_tokenize

def process_text(text: str, filename: str):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    idx = 0

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:
            current_chunk += " " + sentence
        else:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": current_chunk.strip(),
                "source": filename,
                "index": idx
            })
            idx += 1
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": current_chunk.strip(),
            "source": filename,
            "index": idx
        })

    return chunks
