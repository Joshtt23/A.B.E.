from typing import List
import torch


class Config:
    ACCEPT_LANGUAGE: str = "en-US,en;q=0.9"
    API_KEY: str = "1d5cf60e30a349a08453563a05adbc1d"
    BATCH_SIZE: int = 16
    DATA_CSV: str = "C:\\Users\\josht\\Desktop\\News ML\\data.csv"
    DEVICE_CUDA: int = 0 if torch.cuda.is_available() else -1
    EXCLUDE_LIST: List[str] = [
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
    ENTITY_GROUPS: List[str] = [
        "STOCK", 
        "BUSINESS", 
        "CRYPTO", 
        "PERSON"
    ]
    KEYWORDS_LIMIT: int = 10
    KEYWORD_EXTRACTION_AVERAGE_STRATEGY: str = "macro"
    LEARNING_RATE: float = 1e-05
    LOG_LEVEL: str = "INFO"
    MAX_SECTION_LENGTH: int = 512
    MAX_SUMMARY_LENGTH: int = 80
    MAX_WORKERS: int = 4
    MIN_SUMMARY_LENGTH: int = 30
    NEGATIVE_THRESHOLD: float = -0.2
    NER_MODEL: str = "dbmdz/bert-base-cased-finetuned-conll03-english"
    NEWS_COUNT: int = 10
    NUM_EPOCHS: int = 10
    POSITIVE_THRESHOLD: float = 0.2
    RAKE_MAX_LENGTH: int = 3
    RAKE_MIN_LENGTH: int = 1
    RAKE_STOPWORDS: str = "None"
    SAVED_MODEL_DIR: str = "saved_model"
    SEARCH_TERM: str = "crypto stock market news"
    SENTIMENT_ANALYSIS_MODEL: str = "bert-base-uncased"
    SENTIMENT_AVERAGE_STRATEGY: str = "macro"
    SENTIMENT_LABELS: List[str] = ["Negative", "Positive", "Neutral"]
    SERVER_INTERVAL: int = 60
    SPACY_MODEL: str = "en_core_web_trf"
    SPACY_POS_TAGS: List[str] = [
        'NOUN', 
        'PROPN'
    ]
    SUMMARY_GENERATION_AVERAGE_STRATEGY: str = "macro"
    SUMMARY_MODEL: str = "t5-small"
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
    YAKE_LANGUAGE: str = "en"
    YAKE_MAX_NGRAMS: int = 3
