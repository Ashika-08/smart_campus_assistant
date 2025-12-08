import os
import fitz  # PyMuPDF for PDF extraction
from pptx import Presentation
from docx import Document
import pytesseract
from PIL import Image

# -------------------------------
# Extract text from PDF
# -------------------------------

def extract_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text


# -------------------------------
# Extract text from PPTX
# -------------------------------

def extract_pptx(path):
    text = ""
    prs = Presentation(path)
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text


# -------------------------------
# Extract text from DOCX
# -------------------------------

def extract_docx(path):
    text = ""
    doc = Document(path)
    for p in doc.paragraphs:
        text += p.text + "\n"
    return text


# -------------------------------
# Extract text from Images → OCR
# -------------------------------

def extract_image(path):
    try:
        image = Image.open(path)
        text = pytesseract.image_to_string(image)
        return text
    except:
        return ""


# -------------------------------
# AUTO-DETECT FILE TYPE
# -------------------------------

def extract_text_from_file(path):
    """
    Automatically detect the file type and extract text.
    """

    if not os.path.exists(path):
        return ""

    ext = path.lower()

    if ext.endswith(".pdf"):
        return extract_pdf(path)

    elif ext.endswith(".pptx"):
        return extract_pptx(path)

    elif ext.endswith(".docx"):
        return extract_docx(path)

    elif ext.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        return extract_image(path)

    else:
        # Fallback: Read as plain text
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except:
            return ""
