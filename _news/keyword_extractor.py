from rake_nltk import Rake
import spacy
from yake import KeywordExtractor as YakeKeywordExtractor
from summa import keywords as summa_keyword_extractor
from config import Config
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging


class KeywordExtractor:
    def __init__(self):
        self.rake = Rake(
            min_length=Config.RAKE_MIN_LENGTH,
            max_length=Config.RAKE_MAX_LENGTH,
            stopwords=Config.RAKE_STOPWORDS,
        )
        self.nlp = spacy.load(Config.SPACY_MODEL)
        self.yake = YakeKeywordExtractor(
            lan=Config.YAKE_LANGUAGE, n=Config.YAKE_MAX_NGRAMS
        )
        tokenizer = AutoTokenizer.from_pretrained(Config.NER_MODEL)
        model = AutoModelForTokenClassification.from_pretrained(Config.NER_MODEL)
        self.transformer = pipeline(
            "ner", model=model, tokenizer=tokenizer, device=Config.DEVICE_CUDA
        )
        self.logger = logging.getLogger(__name__)

    def extract_keywords_rake(self, text):
        self.rake.extract_keywords_from_text(text)
        keywords = self.rake.get_ranked_phrases()[: Config.KEYWORDS_LIMIT]
        self.logger.info("Rake keyword extraction completed.")
        return keywords

    def extract_keywords_spacy(self, text):
        doc = self.nlp(text)
        keywords = [
            token.text
            for token in doc
            if not token.is_stop and token.pos_ in Config.SPACY_POS_TAGS
        ][: Config.KEYWORDS_LIMIT]
        self.logger.info("Spacy keyword extraction completed.")
        return keywords

    def extract_keywords_huggingface(self, text):
        # Extract keywords using the Hugging Face model
        results = self.transformer(text)
        keywords = [
            entity["word"]
            for entity in results
            if entity["entity"] == Config.HUGGINGFACE_ENTITY_TYPE
        ][: Config.KEYWORDS_LIMIT]
        self.logger.info("Hugging Face keyword extraction completed.")
        return keywords

    def extract_keywords_yake(self, text):
        keywords = self.yake.extract_keywords(text)
        keywords = [keyword[0] for keyword in keywords][: Config.KEYWORDS_LIMIT]
        self.logger.info("Yake keyword extraction completed.")
        return keywords

    def extract_keywords_textrank(self, text):
        keywords = summa_keyword_extractor.keywords(text, split=True)[
            : Config.KEYWORDS_LIMIT
        ]
        self.logger.info("Textrank keyword extraction completed.")
        return keywords
