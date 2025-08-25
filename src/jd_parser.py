import fitz
import docx

def extract_text_from_jd(file_path):
    """Extract text from a job description file."""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
