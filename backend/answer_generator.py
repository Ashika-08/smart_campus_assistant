import os
import json
from llm_client import call_llm

def generate_answer(query, results):
    
    
    context_parts = []
    for i, res in enumerate(results[:5]): 
        source = res.get("source", "unknown")
        text = res.get("text", "").strip()
        context_parts.append(f"Source {i+1} ({source}):\n{text}")
    
    context_str = "\n\n".join(context_parts)
    
    if not context_str:
        return "I could not find any relevant information in the documents to answer your question."

    prompt = f"""You are a helpful academic assistant. Answer the user's question based ONLY on the context provided below.
    
    Citation Rule:
    - Every claim must be followed by a citation in the format [Source N] if derived from context.
    - Example: "Machine learning is useful for agriculture [Source 1]."
    
    Hybrid Knowledge Rule:
    - Prioritize context.
    - CRITICAL: If the context is empty, irrelevant, or talks about a different topic (e.g. Psychology NLP vs AI NLP), YOU MUST USE YOUR GENERAL KNOWLEDGE.
    - Use your internal knowledge to answer the question fully.
    - Start general knowledge answers with "Based on general knowledge...".
    
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
