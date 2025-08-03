# src/summarization.py

from transformers import T5Tokenizer, T5ForConditionalGeneration

class LegalDocumentSummarizer:
    def __init__(self, model_name='t5-small'):
        print("[INFO] Loading T5-small model...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def summarize(self, text, max_length=150, min_length=40):
        """Summarize long legal documents using pre-trained T5."""
        input_text = "summarize: " + text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        summary_ids = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == "__main__":
    summarizer = LegalDocumentSummarizer()

    sample_text = """
    This Agreement is entered into between ABC Corp and XYZ Ltd.
    The parties agree to maintain confidentiality and follow the terms set forth herein.
    Payment shall be made in full within 60 days of receipt of invoice.
    In case of dispute, governing law will be that of New York.
    """

    summary = summarizer.summarize(sample_text)
    print("\n[SUMMARY]:")
    print(summary)
