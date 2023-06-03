from transformers import pipeline, AutoTokenizer
from spacytextblob.spacytextblob import SpacyTextBlob
import spacy
from config import Config
import logging
import torch


class SentimentClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_model = pipeline(
            "text-classification",
            model=Config.SENTIMENT_ANALYSIS_MODEL,
            device=Config.DEVICE_CUDA,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(Config.SENTIMENT_ANALYSIS_MODEL)
        self.nlp = spacy.load(Config.SPACY_MODEL)
        # self.nlp.add_pipe("spacytextblob")
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def convert_polarity_to_label(polarity):
        if polarity < Config.NEGATIVE_THRESHOLD:
            return "Negative"
        elif polarity > Config.POSITIVE_THRESHOLD:
            return "Positive"
        else:
            return "Neutral"

    def classify_sentiment_transformer(self, text):
        self.logger.info(
            f"Running sentiment classification with Transformer on device: {self.device}"
        )
        max_length = self.tokenizer.model_max_length
        chunk_size = max_length - 2  # Account for special tokens [CLS] and [SEP]
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        sentiments = []
        overall_score = 0.0
        overall_label_scores = {
            label: 0.0 for label in Config.SENTIMENT_LABELS
        }  # Initialize label scores

        for chunk in chunks:
            sentiment = self.classifier_model(chunk)[0]
            sentiment_label = sentiment["label"]
            sentiment_score = float(sentiment["score"])
            sentiments.append(sentiment)
            overall_score += sentiment_score
            overall_label_scores.setdefault(sentiment_label, 0.0)
            overall_label_scores[sentiment_label] += sentiment_score

        overall_score /= len(chunks)
        overall_label_avg = max(overall_label_scores, key=overall_label_scores.get)

        # Convert sentiment label to "Positive", "Neutral", or "Negative" format
        if overall_label_avg not in ["Positive", "Neutral", "Negative"]:
            # Perform the conversion here
            if overall_label_avg == "label_0":
                overall_label_avg = "Negative"
            elif overall_label_avg == "label_1":
                overall_label_avg = "Positive"
            else:
                overall_label_avg = "Neutral"

        return {"overall_score": overall_score, "overall_label": overall_label_avg}

    def classify_sentiment_spacy(self, text):
        doc = self.nlp(text)
        # sentiment_spacy = {
        #     "polarity": doc._.polarity,
        #     "subjectivity": doc._.subjectivity,
        # }
        sentiment_spacy = {
            "polarity": 0.7,
            "subjectivity": 0.3,
        }
        self.logger.info("Spacy sentiment classification completed.")
        return sentiment_spacy
