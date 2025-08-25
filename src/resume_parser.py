import fitz  # PyMuPDF for PDF parsing
import docx
import io

def extract_resume_text(uploaded_file):
    """
    Extract text from an uploaded resume (PDF or DOCX).
    
    Parameters:
        uploaded_file (UploadedFile): The resume file uploaded via Streamlit.
    
    Returns:
        str: Extracted text from the resume.
    """
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        return extract_text_from_docx(uploaded_file)
    else:
        return "⚠ Unsupported file format. Please upload a PDF or DOCX file."

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF file using PyMuPDF.
    
    Parameters:
        uploaded_file (UploadedFile): The uploaded PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        return f"⚠ Error extracting text from PDF: {e}"
    
    return text.strip()

def extract_text_from_docx(uploaded_file):
    """
    Extract text from a DOCX file using python-docx.
    
    Parameters:
        uploaded_file (UploadedFile): The uploaded DOCX file.
    
    Returns:
        str: Extracted text from the DOCX file.
    """
    try:
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"⚠ Error extracting text from DOCX: {e}"
    
    return text.strip()
