# src/clause_extraction.py

import spacy

class ClauseExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        """Extract and relabel named entities."""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            label = ent.label_

            # Re-label common legal types
            if "section" in ent.text.lower() or "ipc" in ent.text.lower():
                label = "LAW"
            elif "court" in ent.text.lower():
                label = "COURT"
            elif "police station" in ent.text.lower():
                label = "POLICE STATION"

            entities.append((ent.text.strip(), label))
        return entities

    def extract_key_clauses(self, text, keywords=None):
        """Extract key legal clauses using keyword search."""
        if keywords is None:
            keywords = [
                "termination", "payment", "confidentiality", "governing law",
                "obligation", "bail", "apprehend arrest", "falsely implicated",
                "furnish bail", "protection under section", "cooperate with investigation",
                "not tamper with evidence", "interest of justice", "hon’ble court",
                "this agreement", "license", "disclosure", "liability", "settlement", "trust"
            ]

        doc = self.nlp(text)
        clauses = []

        for sent in doc.sents:
            if any(keyword.lower() in sent.text.lower() for keyword in keywords):
                clauses.append(sent.text.strip())

        return clauses


# Test example
if __name__ == "__main__":
    extractor = ClauseExtractor()
    sample_text = """
    The applicant apprehends arrest in the said matter and seeks protection under Section 438 of the Code of Criminal Procedure.
    The agreement shall be governed by the laws of India.
    Payment shall be made within 30 days of invoice date.
    Either party may terminate this agreement by giving 30 days written notice.
    Confidentiality must be maintained regarding sensitive data.
    """

    print("Named Entities:")
    for ent, label in extractor.extract_entities(sample_text):
        print(f"{ent} [{label}]")

    print("\nKey Clauses:")
    for clause in extractor.extract_key_clauses(sample_text):
        print(f"- {clause}")
