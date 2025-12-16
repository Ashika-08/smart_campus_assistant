import spacy
import networkx as nx
import re

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def _normalize_entity_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    s = name.replace("\n"," ").strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.strip(" :;,.–—-")
    endings = ["application","application:","application."]
    for end in endings:
        if s.lower().endswith(end):
            s = s[:-(len(end))].strip()
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
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        nouns = []
        for nc in doc.noun_chunks:
            t = nc.text.strip()
            if len(t.split())<=6:
                nouns.append((t,"NOUN_CHUNK"))
        combined = ents + nouns
        uniq = []
        seen=set()
        for t,l in combined:
            key = _normalize_entity_name(t)
           
            if not key or len(key) < 3 or re.match(r'^[\d\W]+$', key):
                continue
                
            if key.lower() not in seen:
                seen.add(key.lower())
                uniq.append((key,l))
        return uniq

    def extract_relations(self, text: str):
        doc = nlp(text)
        triples=[]
        for sent in doc.sents:
            ents = [ent.text for ent in sent.ents]
            if len(ents)>=2:
                s,o = ents[0], ents[1]
                triples.append((s,"related_to",o))
        return triples

    def build_from_chunks(self,chunks:list,source_name="unknown"):
        for ch in chunks:
            text = ch.get("text","")
            chunk_label=f"{source_name}_chunk_{ch['index']}"
            self.add_chunk_node(chunk_label, chunk_index=ch["index"], source=source_name)
            
            ents=self.extract_entities(text)
            for ent_text, ent_label in ents:
                ent_norm=_normalize_entity_name(ent_text)
                if ent_norm:
                    self.add_entity_node(ent_norm, label=ent_label)
                    self.add_relation(chunk_label, ent_norm, "mentions")
            rels = self.extract_relations(text)
            for s,r,o in rels:
                s_norm = _normalize_entity_name(s)
                o_norm = _normalize_entity_name(o)
                if s_norm and o_norm:
                    self.add_entity_node(s_norm)
                    self.add_entity_node(o_norm)
                    self.add_relation(s_norm,o_norm,r)

    def to_dict(self):
        return {"nodes":[{"id":n, **self.g.nodes[n]} for n in self.g.nodes],"edges":[{"source":u,"target":v, **self.g.edges[u,v]} for u,v in self.g.edges]}

    def get_subgraph(self, top_k: int = None):
        if not top_k or top_k >= len(self.g.nodes):
            return self.to_dict()
        
       
        centrality = nx.degree_centrality(self.g)
        
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, c in sorted_nodes[:top_k]]
        
        
        subg = self.g.subgraph(top_nodes)
        
        return {
            "nodes": [{"id": n, **subg.nodes[n]} for n in subg.nodes],
            "edges": [{"source": u, "target": v, **subg.edges[u,v]} for u, v in subg.edges]
        }

_KG = KnowledgeGraph()
_KG_STORAGE = {"default": _KG}

def get_kg():
    global _KG
    return _KG

def build_and_save_kg(chunks, source_name="unknown"):
    global _KG
    _KG = KnowledgeGraph()
    _KG.build_from_chunks(chunks, source_name=source_name)
    _KG_STORAGE["default"] = _KG
    return _KG.to_dict()

def expand_graph(entity_name: str, hops: int = 1):
    kg = get_kg()
    if not kg.g.has_node(entity_name):
        return []
    visited = set([entity_name])
    frontier = set([entity_name])
    for _ in range(hops):
        new_frontier = set()
        for node in frontier:
            new_frontier.update(list(kg.g.successors(node)))
            new_frontier.update(list(kg.g.predecessors(node)))
        frontier = new_frontier
        visited.update(new_frontier)
    visited.discard(entity_name)
    return list(visited)
