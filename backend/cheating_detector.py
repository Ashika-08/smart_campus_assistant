# cheating_detector.py (Robust + No False Positives Version)

import re
import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------
# Utility: normalize & lemmatize words
# -------------------------------------------
def normalize(text):
    doc = nlp(text.lower())
    words = []
    for token in doc:
        if token.is_stop or token.is_punct or len(token.lemma_) <= 2:
            continue
        words.append(token.lemma_)
    return set(words)

# -------------------------------------------
# Embedding similarity (semantic alignment)
# -------------------------------------------
def semantic_alignment(answer_text, context_text):
    if not answer_text.strip() or not context_text.strip():
        return 0

    emb_answer = encoder.encode(answer_text, convert_to_tensor=True)
    emb_context = encoder.encode(context_text, convert_to_tensor=True)

    sim = float(util.cos_sim(emb_answer, emb_context))
    return sim

# -------------------------------------------
# Main cheating detector
# -------------------------------------------
def is_cheating(question, context_chunks, llm_answer):
    context_text = " ".join([c["text"] for c in context_chunks])

    # Extract keywords by normalization
    q_terms = normalize(question)
    a_terms = normalize(llm_answer)
    c_terms = normalize(context_text)

    # ---------------------------------------
    # 1) Question concepts missing from context?
    # Only suspicious if ALL missing.
    # ---------------------------------------
    missing = [t for t in q_terms if t not in c_terms]

    if len(missing) == len(q_terms):  # only if ALL missing → cheating
        return True

    # ---------------------------------------
    # 2) Answer hallucinating concepts?
    # ---------------------------------------
    hallucinated = [t for t in a_terms if t not in c_terms]

    if len(hallucinated) > 3:  # allow small drift
        return True

    # ---------------------------------------
    # 3) Numeric hallucination
    # ---------------------------------------
    nums_answer = set(re.findall(r"\d+", llm_answer))
    nums_context = set(re.findall(r"\d+", context_text))

    if nums_answer - nums_context:
        return True

    # ---------------------------------------
    # 4) Semantic similarity check
    # ---------------------------------------
    sim = semantic_alignment(llm_answer, context_text)
    if sim < 0.2:  # low threshold = more tolerant
        return True

    return False
