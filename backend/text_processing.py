import uuid
from nltk.tokenize import sent_tokenize

def process_text(text, filename):
    """
    Splits extracted text into chunks + adds metadata including required 'index'.
    """

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    index = 0   

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:
            current_chunk += " " + sentence
        else:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": current_chunk.strip(),
                "source": filename,
                "index": index           
            })
            current_chunk = sentence
            index += 1

    
    if current_chunk.strip():
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": current_chunk.strip(),
            "source": filename,
            "index": index              
        })

    return chunks
