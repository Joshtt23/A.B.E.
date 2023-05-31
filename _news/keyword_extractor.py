from rake_nltk import Rake
import spacy
from yake import KeywordExtractor as YakeKeywordExtractor
from summa import keywords as summa_keyword_extractor
from config import Config
import torch
from transformers import pipeline

class KeywordExtractor:
    def __init__(self):
        self.rake = Rake(min_length=Config.RAKE_MIN_LENGTH, max_length=Config.RAKE_MAX_LENGTH, stopwords=Config.RAKE_STOPWORDS)
        self.nlp = spacy.load(Config.SPACY_MODEL)
        self.yake = YakeKeywordExtractor(lan=Config.YAKE_LANGUAGE, n=Config.YAKE_MAX_NGRAMS)

    def extract_keywords_rake(self, text):
        self.rake.extract_keywords_from_text(text)
        keywords = self.rake.get_ranked_phrases()[:Config.KEYWORDS_LIMIT]
        return keywords

    def extract_keywords_spacy(self, text):
        doc = self.nlp(text)
        keywords = [token.text for token in doc if not token.is_stop and token.pos_ in Config.SPACY_POS_TAGS][:Config.KEYWORDS_LIMIT]
        return keywords

    def extract_keywords_huggingface(self, text):
        # Initialize the keyword extraction pipeline
        keyword_pipeline = pipeline(
            "ner", 
            model=Config.NER_MODEL, 
            tokenizer=Config.NER_MODEL,
            device=Config.NER_DEVICE
        )
        
        # Extract keywords using the Hugging Face model
        results = keyword_pipeline(text)
        keywords = [entity["word"] for entity in results if entity["entity"] == Config.HUGGINGFACE_ENTITY_TYPE][:Config.KEYWORDS_LIMIT]
        
        return keywords

    def extract_keywords_yake(self, text):
        keywords = self.yake.extract_keywords(text)
        keywords = [keyword[0] for keyword in keywords][:Config.KEYWORDS_LIMIT]
        return keywords

    def extract_keywords_textrank(self, text):
        keywords = summa_keyword_extractor.keywords(text, split=True)[:Config.KEYWORDS_LIMIT]
        return keywords
