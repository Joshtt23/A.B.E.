# config.py
import torch

class Config:

    RAKE_MIN_LENGTH = 1  # Specify the minimum word length for RAKE algorithm
    RAKE_MAX_LENGTH = 3  # Specify the maximum word length for RAKE algorithm
    RAKE_STOPWORDS = None  # Specify the stopwords list for RAKE algorithm
    NEGATIVE_THRESHOLD = -0.2  # Define the negative polarity threshold for sentiment classification
    POSITIVE_THRESHOLD = 0.2  # Define the positive polarity threshold for sentiment classification
    SENTIMENT_LABELS = ["Negative", "Positive", "Neutral"]  # Define the possible sentiment labels
    YAKE_LANGUAGE = "en"  # Specify the language for YAKE algorithm
    YAKE_MAX_NGRAMS = 3  # Specify the maximum number of n-grams for YAKE algorithm
    API_KEY = '1d5cf60e30a349a08453563a05adbc1d'
    SEARCH_TERM = 'cryptocurrency stock market news' # Define Bing Search Term
    SENTIMENT_ANALYSIS_MODEL = "roberta-large-mnli" # Transformer model for sentiment analysis
    NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english" # Transformer model for NER
    SUMMARY_MODEL = "t5-large" # Transformer model for summary generation
    SPACY_POS_TAGS = ["NOUN", "PROPN"]  # Specify the desired POS tags for Spacy keyword extraction
    NER_DEVICE = 0 if torch.cuda.is_available() else -1  # Specify the device for Hugging Face NER model
    HUGGINGFACE_ENTITY_TYPE = "MISC"  # Specify the desired entity type for Hugging Face keyword extraction
    KEYWORDS_LIMIT = 10  # Specify the maximum number of keywords to extract
    LOG_LEVEL = "INFO" # Specify the logging level
    MAX_LENGTH = 80 # Specify the maximum length of the input text for summarization
    MIN_LENGTH = 30 # Specify the minimum length of the input text for summarization
    TARGET_LENGTH_RATIO = 0.3 # Specify the target length ratio for summarization
    MAX_SUMMARY_LENGTH = 25 # Specify the maximum length of the summary
    MAX_SECTION_LENGTH = 1024   # Specify the maximum length of the input text for summarization
    MAX_WORKERS = 4  # Set the desired number of maximum workers for ThreadPoolExecutor
    SPACY_MODEL = "en_core_web_trf" # Specify the Spacy model
    NUM_SENTENCES = 3 # Specify the number of sentences for summarization
    NEWS_COUNT = 100 # Specify the number of news articles to fetch
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
    SENTIMENT_AVERAGE_STRATEGY = "macro" # Sentiment analysis configuration
    KEYWORD_EXTRACTION_AVERAGE_STRATEGY = "macro" # Keyword extraction configuration
    SUMMARY_GENERATION_AVERAGE_STRATEGY = "macro" # Summary generation configuration