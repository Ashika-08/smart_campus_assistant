
import os
import subprocess


def call_ollama(prompt: str, model: str = "llama3"):
    """
    Calls Ollama using Python API if available, otherwise uses CLI fallback.
    Returns { "answer": str, "meta": {...} }
    """

    
    try:
        import ollama
        response = ollama.generate(
            model=model,
            prompt=prompt
        )
        answer = response.get("response", "").strip()
        return {
            "answer": answer,
            "meta": {"provider": "ollama", "raw": response}
        }
    except Exception as e1:
       
        try:
            process = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=120
            )
            answer = process.stdout.strip()
            return {
                "answer": answer,
                "meta": {"provider": "ollama_cli", "returncode": process.returncode}
            }
        except Exception as e2:
            raise RuntimeError(f"Ollama failed: {e1}\nCLI failed: {e2}")



def call_openai(prompt: str, model: str = "gpt-4o-mini"):
    """
    Calls OpenAI API only if OPENAI_API_KEY exists.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer ONLY using the given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        answer = resp.choices[0].message.content
        return {
            "answer": answer,
            "meta": {"provider": "openai", "raw": resp}
        }

    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")


def call_llm(prompt: str, model: str = None):
    """
    Central LLM calling function.
    Tries Ollama → then OpenAI (if key exists) → else throws error.
    """

    model = model or os.getenv("LLM_MODEL", "llama3")

    
    try:
        return call_ollama(prompt, model=model)
    except Exception as ollama_error:
        print("Ollama failed →", ollama_error)

        
        if os.getenv("OPENAI_API_KEY"):
            try:
                return call_openai(prompt, model=model)
            except Exception as openai_error:
                raise RuntimeError(
                    f"Ollama failed and OpenAI also failed.\n\n"
                    f"Ollama error:\n{ollama_error}\n\n"
                    f"OpenAI error:\n{openai_error}"
                )

        
        raise RuntimeError(
            f"Ollama failed AND no OpenAI API key configured.\n\n"
            f"Ollama error:\n{ollama_error}"
        )
