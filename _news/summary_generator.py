from transformers import pipeline
import spacy
from config import Config

class SummaryGenerator:
    def __init__(self):
        self.transformer_summarizer = pipeline('summarization', model=Config.SUMMARY_MODEL)
        self.nlp = spacy.load(Config.SPACY_MODEL)

    def generate_summary(self, text, target_length=None, max_summary_length=None):
        input_length = len(text)
        target_length = target_length or Config.TARGET_LENGTH_RATIO
        max_summary_length = max_summary_length or Config.MAX_SUMMARY_LENGTH

        if input_length <= Config.MAX_SECTION_LENGTH:
            max_length = min(int(input_length * target_length), max_summary_length)
            summary = self.transformer_summarizer(text, max_length=max_length, min_length=max_length, do_sample=False)[0]['summary_text']
        else:
            chunks = [text[i:i+Config.MAX_SECTION_LENGTH] for i in range(0, len(text), Config.MAX_SECTION_LENGTH)]
            summaries = []
            for chunk in chunks:
                chunk_summary = self.transformer_summarizer(chunk, max_length=max_summary_length, min_length=max_summary_length, do_sample=False)[0]['summary_text']
                summaries.append(chunk_summary)
            summary = " ".join(summaries)

        return summary

    def generate_spacy(self, text, num_sentences=None):
        doc = self.nlp(text)
        num_sentences = num_sentences or Config.NUM_SENTENCES
        sentences = [sent.text for sent in doc.sents]
        summary = " ".join(sentences[:num_sentences])  # Generate summary using the specified number of sentences
        return summary

    def generate_gensim(self, text, ratio=0.3):
        # summary = gensim_summarize(text, ratio=ratio)
        summary = "NOT IMPLEMENTED YET"
        return summary
