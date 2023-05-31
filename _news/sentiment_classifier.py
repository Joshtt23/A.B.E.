from transformers import pipeline, AutoTokenizer
from spacytextblob.spacytextblob import SpacyTextBlob
import spacy
from config import Config

class SentimentClassifier:
    def __init__(self):
        self.classifier_model = pipeline('sentiment-analysis', model=Config.SENTIMENT_ANALYSIS_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.SENTIMENT_ANALYSIS_MODEL)
        self.nlp = spacy.load(Config.SPACY_MODEL)
        self.nlp.add_pipe('spacytextblob')

    @staticmethod
    def convert_polarity_to_label(polarity):
        if polarity < Config.NEGATIVE_THRESHOLD:
            return 'Negative'
        elif polarity > Config.POSITIVE_THRESHOLD:
            return 'Positive'
        else:
            return 'Neutral'

    def classify_sentiment_transformer(self, text):
        max_length = self.tokenizer.model_max_length
        chunk_size = max_length - 2  # Account for special tokens [CLS] and [SEP]
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        sentiments = []
        overall_score = 0.0
        overall_label_scores = {label: 0.0 for label in Config.SENTIMENT_LABELS}  # Initialize label scores

        for chunk in chunks:
            sentiment = self.classifier_model(chunk)[0]
            sentiment_label = sentiment['label']
            sentiment_score = float(sentiment['score'])
            sentiments.append(sentiment)
            overall_score += sentiment_score
            overall_label_scores.setdefault(sentiment_label, 0.0)  # Set default score for new label
            overall_label_scores[sentiment_label] += sentiment_score

        overall_score /= len(chunks)
        overall_label_avg = max(overall_label_scores, key=overall_label_scores.get)  # Get label with highest score

        return {
            'overall_score': overall_score,
            'overall_label': overall_label_avg
        }

    def classify_sentiment_spacy(self, text):
        doc = self.nlp(text)
        sentiment_spacy = {
            'polarity': doc._.blob.polarity,
            'subjectivity': doc._.blob.subjectivity,
        }
        return sentiment_spacy
