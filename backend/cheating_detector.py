import re

def detect_cheating(answer: str, chunks: list[str]) -> bool:
    """
    Returns True if cheating is detected (answer contains info not grounded in chunks).
    """

   
    context = " ".join(chunks).lower()
    ans = answer.lower()

    
    important_words = re.findall(r"[a-zA-Z_]{4,}", ans)

    
    for word in important_words:
        if word not in context:
            return True

    
    hallucination_flags = [
        "in general",
        "typically",
        "usually",
        "you can also",
        "another way",
        "for example",
        "let's consider",
        "suppose we have",
        "one approach is",
    ]

    for flag in hallucination_flags:
        if flag in ans and flag not in context:
            return True

    return False
