import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessing import extract_text_from_file, clean_text

class DocumentSimilarity:
    def __init__(self, folder_path="../data/raw_pdfs"):
        self.folder_path = folder_path
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.documents = []
        self.filenames = []
        self.tfidf_matrix = None
        self.load_documents()

    def load_documents(self):
        for category in os.listdir(self.folder_path):
            category_path = os.path.join(self.folder_path, category)
            if os.path.isdir(category_path):
                for file in os.listdir(category_path):
                    if file.endswith(('.pdf', '.docx')):
                        full_path = os.path.join(category_path, file)
                        try:
                            text = extract_text_from_file(full_path)
                            self.documents.append(clean_text(text))
                            self.filenames.append(f"{category}/{file}")
                        except Exception as e:
                            print(f"[WARNING] Could not read {file}: {e}")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def get_similar_documents(self, query_text, top_k=3):
        query_vec = self.vectorizer.transform([clean_text(query_text)])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        return [(self.filenames[i], similarities[i]) for i in top_indices]
