# llm_client.py
import os
import json
import subprocess
import requests

def _call_ollama(prompt: str, model_name: str = "llama3:latest"):
    
    try:
        import ollama
        resp = ollama.chat(model=model_name, messages=[{"role":"user","content":prompt}])
        
        content = resp.get("message", {}).get("content", "")
        return {"answer": content, "meta": {"provider":"ollama", "raw": resp}}
    except Exception:
        
        try:
            p = subprocess.run(["ollama", "chat", model_name, "-m", prompt], capture_output=True, text=True, timeout=60)
            out = p.stdout.strip()
            return {"answer": out, "meta": {"provider":"ollama_cli"}}
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")


def _call_openai(prompt: str, model_name: str = "gpt-4o-mini"):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":"You answer only from context."},{"role":"user","content":prompt}],
        )
        content = resp.choices[0].message["content"]
        return {"answer": content, "meta": {"provider":"openai", "raw": resp}}
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")

def call_llm(prompt: str, model_name: str = None):
    
    model = model_name or os.getenv("LLM_MODEL", "llama3:latest")
    
    try:
        return _call_ollama(prompt, model_name=model)
    except Exception as e1:
        
        if os.getenv("OPENAI_API_KEY"):
            try:
                return _call_openai(prompt, model_name=model)
            except Exception as e2:
                raise RuntimeError(f"Ollama failed: {e1}; OpenAI failed: {e2}")
        else:
            raise RuntimeError(f"Ollama failed and no OPENAI_API_KEY configured. Ollama error: {e1}")
