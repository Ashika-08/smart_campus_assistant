import os
import json
from llm_client import call_llm

def generate_answer(query, results):
    # Prepare context from results
    # Each result has "text", "source", "index", "score"
    # We want to format this nicely for the LLM
    
    context_parts = []
    for i, res in enumerate(results[:5]): # Take top 5
        source = res.get("source", "unknown")
        text = res.get("text", "").strip()
        context_parts.append(f"Source {i+1} ({source}):\n{text}")
    
    context_str = "\n\n".join(context_parts)
    
    if not context_str:
        return "I could not find any relevant information in the documents to answer your question."

    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the context provided below.
    If the answer is not in the context, say "I don't know based on the provided documents."
    
    Context:
    {context_str}
    
    Question: {query}
    
    Answer:
    """
    
    try:
        response = call_llm(prompt)
        return response["answer"]
    except Exception as e:
        print(f"Answer generation error: {e}")
        return "Sorry, I encountered an error while generating the answer."
