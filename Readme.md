# 📚 Legal Document Analyzer: ML + DL + GenAI Powered Web App

An AI-powered system for analyzing legal documents using:

✅ Machine Learning (Document Classification)
✅ NLP (Clause & Entity Extraction, Document Similarity)
✅ Generative AI (Document Summarization)
✅ Streamlit Web UI (Interactive Interface)

---

## 🚀 Demo

Upload legal documents directly via the web interface to:

* 📁 Classify document type
* 📟 Extract key clauses & named entities
* 🤝 Compare document similarity
* 📝 Summarize long legal documents


---

## 🌟 Features

| Module                 | Description                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| 📄 PDF Text Extraction | Extract clean text from uploaded PDFs or Word files                |
| 📚 Document Classifier | ML-based classifier using TF-IDF + RandomForest                    |
| 📁 Clause Extraction   | spaCy-powered NER + keyword-based clause matching                  |
| 🔍 Document Similarity | Cosine similarity over TF-IDF vectors for legal similarity scoring |
| 📝 Document Summarizer | Hugging Face transformer (T5-small) for abstractive summaries      |
| 💻 Streamlit Web UI    | Interactive multi-tab interface                                    |

---

## 📂 Project Structure

```
LegalDocAI/
🔹 data/                  # Raw documents and processed datasets
🔹 models/                # Saved ML models
🔹 src/                   # Source code
├── data_preprocessing.py
├── document_classifier.py
├── clause_extraction.py
├── summarization.py
├── similarity_checker.py
└── ui_app.py          # Streamlit interface
🔹 requirements.txt
🔹 README.md
```

---

## ⚙️ How to Run

1️⃣ Clone & install dependencies:

```bash
git clone https://github.com/your-username/LegalDocAI.git
cd LegalDocAI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2️⃣ Launch the Web UI:

```bash
streamlit run src/ui_app.py
```

---

## 🤖 Tech Stack

* **ML**: scikit-learn (TF-IDF, RandomForest)
* **NLP**: spaCy, NLTK, cosine similarity
* **GenAI**: Hugging Face T5-small
* **UI**: Streamlit

---

## 💼 Why This Project

Legal documents are often long and complex. This tool helps:

* Automate document classification
* Extract meaningful clauses/entities
* Assess similarity between legal texts
* Summarize lengthy contracts in seconds

---

## 🚧 Future Enhancements

* ✅ (Done) Add Legal Document Similarity module
* 🧠 Multi-label classification
* 📟 Clause-wise summaries
* 📅 Download/export annotated results
* 🔍 Fine-tune summarizer on legal datasets (e.g., BillSum)
