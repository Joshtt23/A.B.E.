from transformers import pipeline
import spacy
from config import Config
import logging


class SummaryGenerator:
    def __init__(self):
        self.transformer_summarizer = pipeline(
            "summarization",
            model=Config.SUMMARY_MODEL,
            tokenizer=Config.SUMMARY_MODEL,
            device=Config.DEVICE_CUDA,
        )
        self.nlp = spacy.load(Config.SPACY_MODEL)
        self.logger = logging.getLogger(__name__)

    def generate_summary_transformer(
        self, text, target_ratio=0.3, max_summary_length=None
    ):
        input_length = len(text)
        max_summary_length = max_summary_length or Config.MAX_SUMMARY_LENGTH

        if input_length <= Config.MAX_SECTION_LENGTH:
            max_length = min(int(input_length * target_ratio), max_summary_length)
            summary = self.transformer_summarizer(
                text, max_length=max_length, min_length=max_length, do_sample=False
            )[0]["summary_text"]
        else:
            chunks = [
                text[i : i + Config.MAX_SECTION_LENGTH]
                for i in range(0, len(text), Config.MAX_SECTION_LENGTH)
            ]
            summaries = []
            for chunk in chunks:
                chunk_max_length = min(
                    int(len(chunk) * target_ratio), max_summary_length
                )
                chunk_summary = self.transformer_summarizer(
                    chunk,
                    max_length=chunk_max_length,
                    min_length=chunk_max_length,
                    do_sample=False,
                )[0]["summary_text"]
                summaries.append(chunk_summary)
            summary = " ".join(summaries)

        self.logger.info("Summary generation completed.")
        return summary

    def generate_summary_spacy(self, text):
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        summary = " ".join(
            sentences
        )  # Generate summary using the specified number of sentences
        self.logger.info("Spacy summary generation completed.")
        return summary
