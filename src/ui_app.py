# src/ui_app.py

import os
os.environ["THREADPOOLCTL_SKIP_PYTHON_RUNTIME_THREADPOOL"] = "1"

import streamlit as st
from data_preprocessing import extract_text_from_file, clean_text
from document_classifier import LegalDocumentClassifier
from clause_extraction import ClauseExtractor
from summarization import LegalDocumentSummarizer
from document_similarity import DocumentSimilarity

# Load trained models
classifier = LegalDocumentClassifier()
classifier.load_model("../models/classifier.pkl")

extractor = ClauseExtractor()
summarizer = LegalDocumentSummarizer()
similarity_engine = DocumentSimilarity()

# Streamlit App Setup
st.set_page_config(page_title="Legal Document Analyzer", layout="wide")
st.title("📚 Legal Document Analyzer (ML + DL + GenAI)")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Upload Document", 
    "🏷️ Document Classification", 
    "📑 Clause Extraction", 
    "📝 Document Summary",
    "🧩 Similar Documents"   # New Tab
])


# --- GLOBAL STATE ---
document_text = ""
cleaned_text = ""
predicted_label = None

# --- TAB 1: Upload Document ---
with tab1:
    st.header("Upload Legal Document (PDF or Word)")
    uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "docx"])

    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        temp_file = f"temp_uploaded.{file_ext}"

        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            document_text = extract_text_from_file(temp_file)
            cleaned_text = clean_text(document_text)
            predicted_label = classifier.predict([cleaned_text])[0]

            st.success(f"📑 Predicted Category: **{predicted_label}**")
            st.subheader("📜 Document Preview")
            st.text_area("Extracted Text", document_text[:3000], height=300)

        except Exception as e:
            st.error(f"❌ Error processing document: {e}")

# --- TAB 2: Document Classification ---
with tab2:
    st.header("🏷️ Document Classification")
    if cleaned_text:
        predicted_label = classifier.predict([cleaned_text])[0]
        st.success(f"📌 **Predicted Document Type:** {predicted_label}")
    else:
        st.info("📂 Please upload a document in the first tab.")

# --- TAB 3: Clause Extraction ---
with tab3:
    st.header("📑 Clause & Entity Extraction")
    if document_text:
        try:
            entities = extractor.extract_entities(document_text)
            clauses = extractor.extract_key_clauses(document_text)

            st.subheader("🧠 Named Entities:")
            if entities:
                for ent, label in entities:
                    st.markdown(f"- **{ent}** _({label})_")
            else:
                st.info("No named entities found.")

            st.subheader("🧾 Key Clauses:")
            if clauses:
                for clause in clauses:
                    st.markdown(f"- {clause}")
            else:
                st.info("No key clauses extracted.")
        except Exception as e:
            st.error(f"Error during clause/entity extraction: {e}")
    else:
        st.info("📂 Please upload a document in the first tab.")

# --- TAB 4: Document Summary ---
with tab4:
    st.header("📝 Legal Document Summary")
    if document_text:
        try:
            summary = summarizer.summarize(document_text)
            st.subheader("🔍 Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"Error summarizing document: {e}")
    else:
        st.info("📂 Please upload a document in the first tab.")

# --- TAB 6: Similar Documents ---
with tab5:
    st.header("🧩 Similar Legal Documents")
    if document_text:
        try:
            similar_docs = similarity_engine.get_similar_documents(document_text)
            st.write("Top Similar Documents:")
            for fname, score in similar_docs:
                st.write(f"- 📄 {fname} (Similarity: {score:.2f})")
        except Exception as e:
            st.error(f"Error finding similar documents: {e}")
    else:
        st.info("📂 Please upload a document in the first tab.")

