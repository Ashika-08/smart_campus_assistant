from llm_client import call_llm

def generate_summary(text):
    if not text.strip():
        return "No text available to summarize."
    
    # Limit context to avoid context window issues
    context = text[:10000]
    
    prompt = f"Please provide a concise summary of the following text:\n\n{context}"
    
    try:
        response = call_llm(prompt)
        return response["answer"]
    except Exception as e:
        return f"Error generating summary: {str(e)}"
