# llm_client.py
import os
import json
import subprocess
import requests

# Option 1: Ollama local (preferred if you installed it)
def _call_ollama(prompt: str, model_name: str = "llama3:latest"):
    # uses the ollama python client if installed; else call the CLI
    try:
        import ollama
        resp = ollama.chat(model=model_name, messages=[{"role":"user","content":prompt}])
        # Ollama client returns dictionary like {"message": {"content": "..."}}
        content = resp.get("message", {}).get("content", "")
        return {"answer": content, "meta": {"provider":"ollama", "raw": resp}}
    except Exception:
        # fallback to CLI
        try:
            p = subprocess.run(["ollama", "chat", model_name, "-m", prompt], capture_output=True, text=True, timeout=60)
            out = p.stdout.strip()
            return {"answer": out, "meta": {"provider":"ollama_cli"}}
        except Exception as e:
            raise RuntimeError(f"Ollama call failed: {e}")

# Option 2: OpenAI (if you prefer)
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
    # Choose order: Ollama local -> OpenAI
    model = model_name or os.getenv("LLM_MODEL", "llama3:latest")
    # Try Ollama first
    try:
        return _call_ollama(prompt, model_name=model)
    except Exception as e1:
        # fallback to OpenAI if API key present
        if os.getenv("OPENAI_API_KEY"):
            try:
                return _call_openai(prompt, model_name=model)
            except Exception as e2:
                raise RuntimeError(f"Ollama failed: {e1}; OpenAI failed: {e2}")
        else:
            raise RuntimeError(f"Ollama failed and no OPENAI_API_KEY configured. Ollama error: {e1}")
