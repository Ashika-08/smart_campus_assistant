import nltk
import uuid
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)

def process_text(text, filename):
    """
    Splits extracted text into chunks + adds metadata.
    """

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:
            current_chunk += " " + sentence
        else:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": current_chunk.strip(),
                "source": filename    
            })
            current_chunk = sentence

    
    if current_chunk.strip():
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": current_chunk.strip(),
            "source": filename        # ← IMPORTANT
        })

    return chunks
