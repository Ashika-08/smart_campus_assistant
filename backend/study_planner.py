from llm_client import call_llm
import json
import ast

def generate_study_plan(text, days=5):
    if not text.strip():
        return []
        
    
    context = text[:6000]
    
    prompt = f"""You are an expert academic tutor. Create a structured {days}-Day Study Plan for a student based STRICTLY on the provided syllabus/text.
    
    Requirements:
    1. Output MUST be a valid JSON list of objects.
    2. Each object must have:
       - "day": (integer, e.g., 1)
       - "topic": (string, specific module or concept from text)
       - "activities": (list of strings, e.g., ["Draw a diagram of...", "Summarize key points of...", "Compare X and Y"])
    
    Critical Constraints:
    - Do NOT hallucinate page numbers (like "Read pages 1-5") unless they are explicitly in the text.
    - Do NOT invent book titles not mentioned in the text.
    - Use ACTIVE LEARNING verbs: "Draw", "Analyze", "Compare", "Solve", "Explain", "List".
    - **COMPREHENSIVENESS RULE**: You MUST include EVERY single sub-topic mentioned in the syllabus. Do not skip small details like "two-track models" or "inertial sensors".
    - Distribute the content evenly across the {days} days.
    - If there are many topics, increase the number of activities per day.
    
    Text Context:
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
            plan = json.loads(clean_json)
        except json.JSONDecodeError:
            plan = ast.literal_eval(clean_json)

        return plan
    except Exception as e:
        print(f"Study Plan generation failed: {e}")
        return [{"day": 1, "topic": "Error generating plan", "activities": ["Please try again."]}]
