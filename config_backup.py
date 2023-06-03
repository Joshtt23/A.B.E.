from typing import List

class Config:
    ACCEPT_LANGUAGE: str = "en-US,en;q=0.9"
    API_KEY: str = "1d5cf60e30a349a08453563a05adbc1d"
    DATA_CSV: str = "C:\\Users\\josht\\Desktop\\News ML\\data.csv"
    DEVICE_CUDA: int = 0
    EXCLUDE_LIST: str = "('error', 'access denied', 'forbidden', 'not found', 'bad request', 'unauthorized', 'internal server error', 'service unavailable', 'gateway timeout', '404', '500', '403', '401', '400', '503', '504')"
    HUGGINGFACE_ENTITY_TYPE: str = "MISC"
    KEYWORDS_LIMIT: int = 10
    KEYWORD_EXTRACTION_AVERAGE_STRATEGY: str = "macro"
    LEARNING_RATE: float = 1e-05
    LOG_LEVEL: str = "INFO"
    MAX_LENGTH: int = 80
    MAX_SECTION_LENGTH: int = 512
    MAX_SUMMARY_LENGTH: int = 80
    MAX_WORKERS: int = 4
    MIN_LENGTH: int = 30
    NEGATIVE_THRESHOLD: float = -0.2
    NER_MODEL: str = "dslim/bert-base-NER"
    NEWS_COUNT: int = 10
    NUM_EPOCHS: int = 10
    NUM_SENTENCES: int = 4
    POSITIVE_THRESHOLD: float = 0.2
    RAKE_MAX_LENGTH: int = 3
    RAKE_MIN_LENGTH: int = 1
    RAKE_STOPWORDS: str = "None"
    SAVED_MODEL_DIR: str = "saved_model"
    SEARCH_TERM: str = "crypto stock market news"
    SENTIMENT_ANALYSIS_MODEL: str = "bert-base-uncased"
    SENTIMENT_AVERAGE_STRATEGY: str = "macro"
    SENTIMENT_LABELS: str = "('Negative', 'Positive', 'Neutral')"
    SERVER_INTERVAL: int = 60
    SPACY_MODEL: str = "en_core_web_trf"
    SPACY_POS_TAGS: str = "('NOUN', 'PROPN')"
    SUMMARY_GENERATION_AVERAGE_STRATEGY: str = "macro"
    SUMMARY_MODEL: str = "google/pegasus-xsum"
    TARGET_LENGTH_RATIO: float = 0.4
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
    YAKE_LANGUAGE: str = "en"
    YAKE_MAX_NGRAMS: int = 3
