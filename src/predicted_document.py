# src/predict_document.py

import sys
from data_preprocessing import extract_text_from_file, clean_text
from document_classifier import LegalDocumentClassifier

def predict_label(filepath):
    text = extract_text_from_file(filepath)
    cleaned = clean_text(text)

    model = LegalDocumentClassifier()
    model.load_model("../models/classifier.pkl")

    prediction = model.predict([cleaned])[0]
    print(f"[PREDICTED CATEGORY] → {prediction}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the document file")
    args = parser.parse_args()

    predict_label(args.file)
