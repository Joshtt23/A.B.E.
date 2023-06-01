import torch

class Config:

    RAKE_MIN_LENGTH = 1
    RAKE_MAX_LENGTH = 3
    RAKE_STOPWORDS = None
    NEGATIVE_THRESHOLD = -0.2
    POSITIVE_THRESHOLD = 0.2
    SENTIMENT_LABELS = ["Negative", "Positive", "Neutral"]
    YAKE_LANGUAGE = "en"
    YAKE_MAX_NGRAMS = 3
    API_KEY = '1d5cf60e30a349a08453563a05adbc1d'
    SEARCH_TERM = 'crypto stock market news'
    SENTIMENT_ANALYSIS_MODEL = "roberta-large-mnli"
    NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"  # Transformer model for NER
    SUMMARY_MODEL = "t5-large"  # Transformer model for summary generation
    SPACY_POS_TAGS = ["NOUN", "PROPN"]
    NER_DEVICE = 0 if torch.cuda.is_available() else -1
    HUGGINGFACE_ENTITY_TYPE = "MISC"
    KEYWORDS_LIMIT = 10
    LOG_LEVEL = "INFO"
    MAX_LENGTH = 80
    MIN_LENGTH = 30
    TARGET_LENGTH_RATIO = 0.4
    MAX_SUMMARY_LENGTH = 80
    MAX_SECTION_LENGTH = 512
    MAX_WORKERS = 4
    SPACY_MODEL = "en_core_web_trf"
    NUM_SENTENCES = 4
    NEWS_COUNT = 100
    EXCLUDE_LIST = [
        "error",
        "access denied",
        "forbidden",
        "not found",
        "bad request",
        "unauthorized",
        "internal server error",
        "service unavailable",
        "gateway timeout",
        "404",
        "500",
        "403",
        "401",
        "400",
        "503",
        "504",
    ]
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
    ACCEPT_LANGUAGE = "en-US,en;q=0.9"
    SENTIMENT_AVERAGE_STRATEGY = "macro"
    KEYWORD_EXTRACTION_AVERAGE_STRATEGY = "macro"
    SUMMARY_GENERATION_AVERAGE_STRATEGY = "macro"
    TRAIN_DATA_CSV = 'tnt/data.csv'
    SAVED_MODEL_DIR = 'saved_model'
    LEARNING_RATE = 1e-5
