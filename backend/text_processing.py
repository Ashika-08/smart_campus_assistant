import re
import uuid
import nltk
from nltk.tokenize import sent_tokenize


def clean_text(text):
    if not text:
        return ""

    
    text = re.sub(r"\n+", "\n", text)

   
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

   
    text = re.sub(r"\s+", " ", text)

    
    text = text.strip()

    return text



def split_into_sentences(text):
    return sent_tokenize(text)



def chunk_text(sentences, max_words=500):
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        
        if word_count + sentence_words > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0

        current_chunk.append(sentence)
        word_count += sentence_words

   
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



def process_text(text, filename="unknown"):
    cleaned = clean_text(text)
    sentences = split_into_sentences(cleaned)
    chunks = chunk_text(sentences)

    processed = []

    for index, chunk in enumerate(chunks):
        entry = {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "index": index,
            "text": chunk,
            "length": len(chunk.split())
        }
        processed.append(entry)

    return processed
