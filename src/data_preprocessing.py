# src/data_preprocessing.py

import os
import re
from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_file(filepath):
    """Extract text from PDF or Word document."""
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError(f"[ERROR] Unsupported file type: {filepath}")

def extract_texts_from_folder(folder_path):
    texts, labels = [], []
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                if file.endswith((".pdf", ".docx")):
                    file_path = os.path.join(category_path, file)
                    try:
                        text = extract_text_from_file(file_path)
                        texts.append(text)
                        labels.append(category)
                    except Exception as e:
                        print(f"[WARNING] Failed to extract {file_path}: {e}")
    return texts, labels

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def clean_all_texts(texts):
    return [clean_text(text) for text in texts]
