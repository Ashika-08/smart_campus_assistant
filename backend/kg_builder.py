

import spacy
import networkx as nx
import re
from difflib import SequenceMatcher
from vector_db import hybrid_retrieve  
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")



def _normalize_entity_name(name: str) -> str:
    """
    Cleans text for KG usage:
      - remove newlines
      - trim spaces / punctuation
      - remove “Application” suffix
    """
    if not isinstance(name, str):
        return str(name)

    s = name.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" :;,.–—-")

    suffixes = ["application", "application:", "application."]
    for suf in suffixes:
        if s.lower().endswith(suf):
            s = s[: -len(suf)].strip()

    return s



class KnowledgeGraph:
    def __init__(self):
        self.g = nx.DiGraph()

    
    def add_entity_node(self, entity_name: str, label: str = "entity", **attrs):
        clean = _normalize_entity_name(entity_name)
        if clean and not self.g.has_node(clean):
            self.g.add_node(clean, label=label, **attrs)

    def add_chunk_node(self, chunk_id: str, **attrs):
        if not self.g.has_node(chunk_id):
            self.g.add_node(chunk_id, label="chunk", **attrs)

  
    def add_relation(self, src: str, dst: str, relation_type: str):
        self.g.add_edge(src, dst, relation=relation_type)

    
    def extract_entities(self, text: str):
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_relations(self, text: str):
        """
        Minimal rule-based relation extraction.
        """
        doc = nlp(text)
        triples = []

        for sent in doc.sents:
            ents = [(ent.text, ent.label_) for ent in sent.ents]
            if len(ents) >= 2:
                s, o = ents[0][0], ents[1][0]
                triples.append((s, "related_to", o))

        return triples

    
    def build_from_chunks(self, chunks: list, source_name="unknown"):
        """
        Build KG from list of chunks.
        """
        for ch in chunks:
            text = ch["text"]
            chunk_label = f"{source_name}_chunk_{ch['index']}"

            self.add_chunk_node(chunk_label, chunk_index=ch["index"], source=source_name)

           
            ents = self.extract_entities(text)
            for ent_text, ent_label in ents:
                ent_norm = _normalize_entity_name(ent_text)
                if ent_norm:
                    self.add_entity_node(ent_norm, label=ent_label)
                    self.add_relation(chunk_label, ent_norm, "mentions")

            
            rels = self.extract_relations(text)
            for s, r, o in rels:
                s_norm = _normalize_entity_name(s)
                o_norm = _normalize_entity_name(o)

                self.add_entity_node(s_norm)
                self.add_entity_node(o_norm)
                self.add_relation(s_norm, o_norm, r)

    
    def to_dict(self):
        return {
            "nodes": [{"id": n, **self.g.nodes[n]} for n in self.g.nodes],
            "edges": [{"source": u, "target": v, **self.g.edges[u, v]} for u, v in self.g.edges],
        }



_KG = KnowledgeGraph()
_KG_STORAGE = {"default": _KG}


def get_kg():
    return _KG


def build_and_save_kg(chunks, source_name="unknown"):
    global _KG
    _KG.build_from_chunks(chunks, source_name=source_name)
    _KG_STORAGE["default"] = _KG
    return _KG.to_dict()



def find_similar_entity_in_kg(query_text: str):
    """
    Fuzzy entity matching between query and KG nodes.
    """
    kg = get_kg()
    nodes = list(kg.g.nodes)

    doc = nlp(query_text)
    candidates = [ent.text for ent in doc.ents]

    if not candidates:
        candidates = [query_text]

    best_match = None
    best_score = 0.0

    for cand in candidates:
        cand_norm = _normalize_entity_name(cand).lower()
        if cand_norm.isdigit() or len(cand_norm) < 3:
            continue

        for node in nodes:
            node_norm = str(node).lower()
            score = SequenceMatcher(None, cand_norm, node_norm).ratio()

            if score > best_score:
                best_score = score
                best_match = node

    if best_score < 0.50:
        return None

    return best_match


def expand_graph(entity_name: str, hops: int = 1):
    """
    BFS graph expansion
    """
    kg = get_kg()

    visited = set([entity_name])
    frontier = set([entity_name])

    for _ in range(hops):
        new_frontier = set()
        for node in frontier:
            new_frontrier = list(kg.g.successors(node)) + list(kg.g.predecessors(node))
            new_frontier.update(new_frontrier)

        frontier = new_frontier
        visited.update(new_frontier)

    visited.remove(entity_name)
    return list(visited)


def hybrid_retrieve_with_graph(query: str, top_k_vector=5, top_k_graph=5, hops=1, alpha=0.6):
    """
    Combines:
      - vector retrieval (dense + sparse)
      - KG entity matching
      - KG hop expansion
    """
    
    vector_hits = hybrid_retrieve(query, merged_k=top_k_vector)

    
    matched_entity = find_similar_entity_in_kg(query)

    if matched_entity is None:
        return {
            "error": f"No similar KG entity found for '{query}'.",
            "vector_results": vector_hits
        }

    
    neighbors = expand_graph(matched_entity, hops=hops)
    neighbors = neighbors[:top_k_graph]

    graph_hits = [{"entity": n, "score": 1.0} for n in neighbors]

    
    return {
        "query": query,
        "matched_entity": matched_entity,
        "vector": vector_hits,
        "graph_expansion": graph_hits
    }
