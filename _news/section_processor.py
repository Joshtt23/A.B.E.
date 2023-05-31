from .keyword_extractor import KeywordExtractor
from .sentiment_classifier import SentimentClassifier
from .summary_generator import SummaryGenerator

class SectionProcessor:
    def __init__(self):
        self.sentiment_classifier = SentimentClassifier()
        self.keyword_extractor = KeywordExtractor()
        self.summary_generator = SummaryGenerator()

    def process_section(self, sec):
        # Sentiment
        sentiment_transformer = self.sentiment_classifier.classify_sentiment_transformer(sec)
        sentiment_spacy, sentiment_label_spacy = self.sentiment_classifier.classify_sentiment_spacy(sec)
        sentiment_spacy.pop('assessments', None)
        sentiment_spacy.pop('ngrams', None)

        # Summary
        summary_spacy = self.summary_generator.generate_summary(sec)

        # Keywords
        keywords_rake = self.keyword_extractor.extract_keywords_rake(sec)[:10]
        keywords_textrank = self.keyword_extractor.extract_keywords_textrank(sec)[:10]
        keywords_spacy = self.keyword_extractor.extract_keywords_spacy(sec)[:10]
        keywords_huggingface = self.keyword_extractor.extract_keywords_huggingface(sec)[:10]
        keywords_yake = self.keyword_extractor.extract_keywords_yake(sec)[:10]

        return {
            'sentiment_transformer': sentiment_transformer,
            'sentiment_spacy': sentiment_spacy,
            'sentiment_label_spacy': sentiment_label_spacy,
            'summary_spacy': summary_spacy,
            'keywords_rake': keywords_rake,
            'keywords_textrank': keywords_textrank,
            'keywords_spacy': keywords_spacy,
            'keywords_huggingface': keywords_huggingface,
            'keywords_yake': keywords_yake,
        }
