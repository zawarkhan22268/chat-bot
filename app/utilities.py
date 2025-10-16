from fastapi import UploadFile, HTTPException
import fitz  # PyMuPDF
from docx import Document
from io import BytesIO
import re

# ---------------- PDF to Text ----------------
def pdf_to_text(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        pdf_bytes = file.file.read()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in pdf])
        return text
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting text from PDF: {str(e)}"
        )

# ---------------- DOCX to Text ----------------
def docs_to_text(file: UploadFile) -> str:
    try:
        file.file.seek(0)
        doc_bytes = file.file.read()
        docx_file = BytesIO(doc_bytes)
        document = Document(docx_file)
        text = "\n".join([p.text for p in document.paragraphs])
        return text
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting text from Word: {str(e)}"
        )

# ---------------- TXT to Text ----------------
def txt_to_text(file: UploadFile) -> str:
    """
    Extracts text content from a .txt file.
    """
    try:
        file.file.seek(0)
        txt_bytes = file.file.read()
        text = txt_bytes.decode("utf-8", errors="ignore")
        return text
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting text from TXT file: {str(e)}"
        )

# ---------------- File Format Detector ----------------
def file_format(file: UploadFile):
    try:
        filename = file.filename.lower()
        if filename.endswith('.pdf'):
            return pdf_to_text(file)
        elif filename.endswith('.docx'):
            return docs_to_text(file)
        elif filename.endswith('.txt'):
            return txt_to_text(file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Only PDF, DOCX, or TXT files are supported."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while processing the file: {str(e)}"
        )

# ---------------- Clean Text ----------------
def clean_text(text: str) -> str:
    """
    Clean text by removing emojis, problematic Unicode characters, and ensuring proper encoding.
    """
    if not text:
        return ""
    
    text = str(text)
    text = re.sub(r'[^\x00-\x7F]', ' ', text)  # remove emojis/non-ASCII
    text = re.sub(r'\s+', ' ', text)           # clean extra spaces/newlines
    text = text.strip()
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")  # ensure UTF-8
    return text
