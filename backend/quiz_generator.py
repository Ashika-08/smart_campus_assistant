from llm_client import call_llm
import json
import re

def generate_quiz(text, num_questions=5):
    if not text.strip():
        return []
        
    context = text[:4000]
    
    prompt = f"""You are a helpful AI assistant. Generate exactly {num_questions} multiple-choice questions based on the provided text.
    
    Requirements:
    1. Output MUST be a valid JSON array of objects.
    2. Each object must have: "question", "options" (list of 4 strings), and "answer" (string).
    3. Do NOT wrap the output in markdown code blocks (e.g., ```json). Return ONLY the raw JSON string.
    4. Ensure the JSON is valid and parsable.

    Text:
    {context}
    """
    
    try:
        response = call_llm(prompt)
        raw_answer = response["answer"]
        raw_answer = response["answer"] 

        
        clean_json = raw_answer.strip().strip("`").replace("json\n", "").strip()
        
        
        start_idx = raw_answer.find("[")
        end_idx = raw_answer.rfind("]")
        
        if start_idx != -1 and end_idx != -1:
            clean_json = raw_answer[start_idx : end_idx + 1]
        else:
            clean_json = raw_answer.strip().strip("`").replace("json\n", "").strip()

        try:
            quiz_data = json.loads(clean_json)
        except json.JSONDecodeError:
            
            import ast
            quiz_data = ast.literal_eval(clean_json)

        return quiz_data
    except Exception as e:
        
        return [{"question": "Could not generate quiz.", "options": ["Error", "Error", "Error", "Error"], "answer": "Error"}]
