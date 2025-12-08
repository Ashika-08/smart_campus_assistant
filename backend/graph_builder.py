# graph_builder.py (Enterprise KG Version)

import os
import json
import spacy

GRAPH_PATH = "graph_store/graph.json"

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# In-memory graph
GRAPH = {}


# ----------------------------------------------
# 1. Load existing graph if available
# ----------------------------------------------
def _load_graph():
    global GRAPH
    if os.path.exists(GRAPH_PATH):
        try:
            with open(GRAPH_PATH, "r", encoding="utf-8") as f:
                GRAPH = json.load(f)
        except:
            GRAPH = {}
    else:
        GRAPH = {}


# ----------------------------------------------
# 2. Save graph to disk
# ----------------------------------------------
def _save_graph():
    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(GRAPH, f, indent=2)


# ----------------------------------------------
# 3. Extract multi-word entities + noun phrases
# ----------------------------------------------
def extract_entities(text):
    doc = nlp(text)

    entities = set()

    # Named entities (ORG, PERSON, etc.)
    for ent in doc.ents:
        cleaned = ent.text.strip()
        if len(cleaned) > 2:
            entities.add(cleaned.lower())

    # Noun chunk extraction
    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip().lower()
        if len(cleaned) > 3:
            entities.add(cleaned)

    return list(entities)


# ----------------------------------------------
# 4. Build Knowledge Graph from chunks
# ----------------------------------------------
def build_graph(chunks):
    """
    GRAPH format:
    GRAPH[entity] = [
        {
            "text": chunk text,
            "source": filename,
            "index": chunk index
        }
    ]
    """

    _load_graph()

    for ch in chunks:
        text = ch["text"]
        source = ch["source"]
        index = ch["index"]

        entities = extract_entities(text)

        for e in entities:
            if e not in GRAPH:
                GRAPH[e] = []

            GRAPH[e].append({
                "text": text,
                "source": source,
                "index": index
            })

            # Deduplicate while preserving order
            uniq = []
            seen = set()
            for item in GRAPH[e]:
                key = (item["source"], item["index"])
                if key not in seen:
                    uniq.append(item)
                    seen.add(key)
            GRAPH[e] = uniq

    _save_graph()


# ----------------------------------------------
# 5. Query knowledge graph with ranking
# ----------------------------------------------
def graph_lookup(query):
    """
    Extract entities from query and return ranked chunks.
    Ranking strategy:
    - Exact entity match = high rank
    - More entities in chunk = higher rank
    """

    _load_graph()

    q_entities = extract_entities(query)
    results = []

    for ent in q_entities:
        if ent in GRAPH:
            for item in GRAPH[ent]:
                score = 1.0
                if ent in item["text"].lower():
                    score += 1.0

                results.append({
                    "text": item["text"],
                    "source": item["source"],
                    "index": item["index"],
                    "score": score
                })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
