import pdfplumber
import os
import pymupdf

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_pdf_pymupdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text

def extract_and_save_text(text_folder, pdf_folder):
    os.makedirs(text_folder, exist_ok=True)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text = extract_text_from_pdf_pymupdf(pdf_path)
            text_file = pdf_file.replace('.pdf', '.txt')
            with open(os.path.join(text_folder, text_file), 'w') as f:
                f.write(text)