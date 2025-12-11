from llm_client import call_llm
import json
import ast

def generate_flashcards(text, num_cards=5):
    if not text.strip():
        return []
        
    context = text[:4000]
    
    prompt = f"""You are a helpful AI assistant. Generate exactly {num_cards} flashcards for students to study the provided text.
    
    Requirements:
    1. Output MUST be a valid JSON array of objects.
    2. Each object must have: "front" (the concept or question) and "back" (the definition or answer).
    3. Do NOT wrap the output in markdown code blocks. Return ONLY the raw JSON string.
    4. Keep definitions concise.

    Text:
    {context}
    """
    
    try:
        response = call_llm(prompt)
        raw_answer = response["answer"]
        
        
        start_idx = raw_answer.find("[")
        end_idx = raw_answer.rfind("]")
        
        if start_idx != -1 and end_idx != -1:
            clean_json = raw_answer[start_idx : end_idx + 1]
        else:
            clean_json = raw_answer.strip().strip("`").replace("json\n", "").strip()

        try:
            cards = json.loads(clean_json)
        except json.JSONDecodeError:
            cards = ast.literal_eval(clean_json)

        return cards
    except Exception as e:
        print(f"Flashcard generation failed: {e}")
        return [{"front": "Error", "back": "Could not generate flashcards."}]
