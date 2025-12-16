from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from vector_db import process_text, add_to_chroma, collection, query_vector_db
from kg_builder import build_and_save_kg, get_kg, expand_graph
from retrieval import hybrid_retrieve_with_graph, hybrid_retrieve
from answer_generator import generate_answer
from quiz_generator import generate_quiz
from summary_generator import generate_summary
from flashcard_generator import generate_flashcards
from study_planner import generate_study_plan

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/plan")
def plan_endpoint(filename: str, days: int = 5):
    all_docs = collection.get(include=["documents", "metadatas"])
    texts = [doc for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]) if meta.get("source") == filename]
    if not texts:
        raise HTTPException(status_code=404, detail="File not found")
    plan = generate_study_plan("\n".join(texts), days)
    return {"filename": filename, "plan": plan}

@app.get("/flashcards")
def flashcards_endpoint(filename: str, num_cards: int = 5):
    all_docs = collection.get(include=["documents", "metadatas"])
    texts = [doc for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]) if meta.get("source") == filename]
    if not texts:
        raise HTTPException(status_code=404, detail="File not found")
    cards = generate_flashcards("\n".join(texts), num_cards)
    return {"filename": filename, "flashcards": cards}



@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    import shutil
    import os
    from extract_text import extract_text

    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        content = extract_text(temp_filename)
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    chunks = process_text(content, file.filename)
    add_to_chroma(chunks)
    return {"status": "added", "chunks": len(chunks)}

@app.get("/process_file")
def process_file(filename: str):
    all_docs = collection.get(include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        if meta.get("source") == filename:
            chunks.append({"text": doc, "source": filename, "index": meta.get("index")})
    if not chunks:
        raise HTTPException(status_code=404, detail="File not found")
    return {"chunks": chunks}



@app.get("/ask")
def ask_question(question: str, top_k: int = 5):
    results = query_vector_db(question, top_k=top_k)
    answer = generate_answer(question, results)
    return {"question": question, "results": results, "answer": answer}

@app.get("/summary")
def summary(filename: str):
    all_docs = collection.get(include=["documents", "metadatas"])
    texts = [doc for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]) if meta.get("source") == filename]
    if not texts:
        raise HTTPException(status_code=404, detail="File not found")
    summary_text = generate_summary("\n".join(texts))
    return {"filename": filename, "summary": summary_text}

@app.get("/quiz")
def quiz(filename: str, num_questions: int = 5):
    all_docs = collection.get(include=["documents", "metadatas"])
    texts = [doc for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]) if meta.get("source") == filename]
    if not texts:
        raise HTTPException(status_code=404, detail="File not found")
    quiz_data = generate_quiz("\n".join(texts), num_questions)
    return {"filename": filename, "quiz": quiz_data}

@app.post("/build_kg")
def build_kg(filename: str):
    all_docs = collection.get(include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        if meta.get("source") == filename:
            chunks.append({"text": doc, "source": filename, "index": meta.get("index")})
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found")
    kg_data = build_and_save_kg(chunks, source_name=filename)
    return {"status": "built", "num_nodes": len(kg_data["nodes"]), "num_edges": len(kg_data["edges"]), "graph": kg_data}

@app.get("/get_graph")
def get_graph_endpoint(limit: int = None):
    kg = get_kg()
    data = kg.get_subgraph(top_k=limit)
    return {"num_nodes": len(data["nodes"]), "num_edges": len(data["edges"]), "graph": data}

@app.get("/kg_query")
def kg_query(entity: str, hops: int = 1):
    kg = get_kg()
    if not kg.g.has_node(entity):
        raise HTTPException(status_code=404, detail=f"Entity '{entity}' not found")
    neighbors = expand_graph(entity, hops=hops)
    nodes = [{"id": n, "label": kg.g.nodes[n].get("label","")} for n in [entity] + neighbors]
    edges = [{"source": u, "target": v, "relation": d.get("relation","")} for u,v,d in kg.g.edges(data=True) if u==entity or v==entity]
    return {"query": entity, "hops": hops, "nodes": nodes, "edges": edges}

@app.get("/hybrid_graph_retrieve")
def hybrid_graph_retrieve(query: str, top_k_vector: int = 5, top_k_graph: int = 5, hops: int = 1, alpha: float = 0.6):
    return hybrid_retrieve_with_graph(query, top_k_vector=top_k_vector, top_k_graph=top_k_graph, hops=hops, alpha=alpha)

@app.get("/hybrid_graph_answer")
def hybrid_graph_answer(query: str, top_k_vector: int = 5, top_k_graph: int = 5, hops: int = 1, alpha: float = 0.6):
    results = hybrid_retrieve_with_graph(query, top_k_vector=top_k_vector, top_k_graph=top_k_graph, hops=hops, alpha=alpha)
    if "vector" not in results:
        return results
    answer = generate_answer(query, results["vector"])
    results["answer"] = answer
    return results
