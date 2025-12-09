from difflib import SequenceMatcher
from kg_builder import get_kg, _normalize_entity_name, nlp
from vector_db import hybrid_retrieve

def find_similar_entity_in_kg(query_text: str, vector_hits: list):
    kg = get_kg()
    nodes = list(kg.g.nodes)
    candidates = []
    doc = nlp(query_text)
    candidates.extend([ent.text for ent in doc.ents])
    candidates.extend([nc.text for nc in doc.noun_chunks])
    for hit in vector_hits:
        doc2 = nlp(hit.get("text",""))
        candidates.extend([ent.text for ent in doc2.ents])
        candidates.extend([nc.text for nc in doc2.noun_chunks])
    if not candidates:
        candidates.append(query_text)
    best_match = None
    best_score = 0.0
    for cand in candidates:
        cand_norm = _normalize_entity_name(cand).lower()
        if not cand_norm or cand_norm.isdigit() or len(cand_norm) < 3:
            continue
        for node in nodes:
            score = SequenceMatcher(None, cand_norm, str(node).lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = node
    if best_score < 0.50:
        return None
    return best_match

def expand_graph(entity_name: str, hops: int = 1):
    from kg_builder import expand_graph as kg_expand
    return kg_expand(entity_name, hops)

def hybrid_retrieve_with_graph(query: str, top_k_vector=5, top_k_graph=5, hops=1, alpha=0.6):
    vector_hits = hybrid_retrieve(query, merged_k=top_k_vector)
    matched_entity = find_similar_entity_in_kg(query, vector_hits)
    if matched_entity is None:
        return {"query": query, "matched_entity": None, "vector": vector_hits, "graph_expansion": []}
    graph_entities = expand_graph(matched_entity, hops=hops)[:top_k_graph]
    graph_hits = [{"entity": e, "score": 1.0} for e in graph_entities]
    return {"query": query, "matched_entity": matched_entity, "vector": vector_hits, "graph_expansion": graph_hits}
