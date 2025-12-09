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

        # Cleanup potential markdown ticks
        clean_json = raw_answer.strip().strip("`").replace("json\n", "").strip()
        
        # specific fix for if it returns valid json wrapped in text
        # Find the first [ and the last ]
        start_idx = raw_answer.find("[")
        end_idx = raw_answer.rfind("]")
        
        if start_idx != -1 and end_idx != -1:
            clean_json = raw_answer[start_idx : end_idx + 1]
        else:
            clean_json = raw_answer.strip().strip("`").replace("json\n", "").strip()

        try:
            quiz_data = json.loads(clean_json)
        except json.JSONDecodeError:
            # Try valid fix - sometimes single quotes are used
            import ast
            quiz_data = ast.literal_eval(clean_json)

        return quiz_data
    except Exception as e:
        print(f"Quiz generation failed: {e}")
        try:
             print(f"Failed Raw Answer: {response['answer']}")
        except:
             pass
        # Fallback to empty or basic error
        return [{"question": "Could not generate quiz. Check server logs.", "options": ["Error"], "answer": "Error"}]
