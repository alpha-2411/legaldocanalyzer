# src/document_classifier.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from data_preprocessing import extract_texts_from_folder, clean_all_texts

class LegalDocumentClassifier:
    def __init__(self):
        self.model = None

    def train(self, texts, labels):
        X = texts
        y = labels

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(solver='liblinear'))
        ])

        self.model.fit(X, y)

    def predict(self, texts):
        return self.model.predict(texts)

    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"[INFO] Model saved to: {path}")

    def load_model(self, path):
        self.model = joblib.load(path)
        print(f"[INFO] Model loaded from: {path}")

if __name__ == "__main__":
    print("[INFO] Loading documents from raw_pdfs...")
    folder_path = "../data/raw_pdfs"
    texts, labels = extract_texts_from_folder(folder_path)

    if not texts:
        print("[ERROR] No texts found.")
        exit()

    cleaned_texts = clean_all_texts(texts)

    model = LegalDocumentClassifier()
    model.train(cleaned_texts, labels)

    model.save_model("../models/classifier.pkl")

    df = pd.DataFrame({"text": cleaned_texts, "label": labels})
    df.to_csv("../data/dataset.csv", index=False)
    print("[INFO] Dataset saved to ../data/dataset.csv")
