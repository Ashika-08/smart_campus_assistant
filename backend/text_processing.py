# text_processing.py (patched enterprise version)

import uuid
import re
from nltk.tokenize import sent_tokenize

CHUNK_SIZE = 600      # Ideal for MiniLM embeddings
CHUNK_OVERLAP = 150   # Provides contextual continuity


# -----------------------------------------------
# Cleaning utilities
# -----------------------------------------------
def clean_text(text: str) -> str:
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix hyphenated line breaks from OCR/PDF
    text = re.sub(r"-\s+", "", text)

    # Strip weird unicode characters
    text = text.replace("\u200b", "").replace("\ufeff", "")

    return text.strip()


# -----------------------------------------------
# Chunking logic (sentences + overlap windows)
# -----------------------------------------------
def process_text(text: str, filename: str):
    """
    Improved chunking:
    - Clean text
    - Split into sentences
    - Build overlapping chunks (600 chars with 150-char overlap)
    """

    text = clean_text(text)
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""
    idx = 0

    for sentence in sentences:
        # If adding sentence doesn't exceed limit → add it
        if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
            current_chunk += " " + sentence
        else:
            # Save chunk
            chunk_text = current_chunk.strip()

            if chunk_text:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "source": filename,
                    "index": idx
                })
                idx += 1

            # Create new chunk starting with overlap from previous chunk
            # Use last 150 chars from previous chunk
            overlap_seed = chunk_text[-CHUNK_OVERLAP:] if len(chunk_text) > CHUNK_OVERLAP else chunk_text

            current_chunk = overlap_seed + " " + sentence

    # Append last chunk
    if current_chunk.strip():
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": current_chunk.strip(),
            "source": filename,
            "index": idx
        })

    return chunks
