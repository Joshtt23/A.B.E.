from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score
import itertools
from config import Config

def flatten_list(nested_list):
    return list(itertools.chain(*nested_list))

def calculate_sentiment_metrics(ground_truth, predictions):    
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average=Config.SENTIMENT_AVERAGE_STRATEGY, zero_division=1)
    recall = recall_score(ground_truth, predictions, average=Config.SENTIMENT_AVERAGE_STRATEGY, zero_division=1)
    f1 = f1_score(ground_truth, predictions, average=Config.SENTIMENT_AVERAGE_STRATEGY, zero_division=1)
    
    return accuracy, precision, recall, f1

def calculate_keyword_extraction_metrics(reference_keywords, extracted_keywords):
    flattened_reference_keywords = flatten_list(reference_keywords)
    flattened_extracted_keywords = flatten_list(extracted_keywords)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        flattened_reference_keywords, flattened_extracted_keywords, average=Config.KEYWORD_EXTRACTION_AVERAGE_STRATEGY, zero_division=1
    )

    return precision, recall, f1

def calculate_summary_generation_metrics(reference_summaries, generated_summaries):
    precision, recall, f1, _ = precision_recall_fscore_support(
        reference_summaries, generated_summaries, average=Config.SUMMARY_GENERATION_AVERAGE_STRATEGY, zero_division=1
    )

    return precision, recall, f1
