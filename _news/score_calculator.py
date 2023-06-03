from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import itertools
from config import Config
from .sentiment_classifier import SentimentClassifier


def flatten_list(nested_list):
    return list(itertools.chain(*nested_list))


def calculate_sentiment_metrics(ground_truth, predictions):
    classifier = SentimentClassifier()
    ground_truth_labels = [
        classifier.convert_polarity_to_label(value) for value in ground_truth
    ]
    predicted_labels = [
        classifier.convert_polarity_to_label(value) for value in predictions
    ]

    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(
        ground_truth_labels,
        predicted_labels,
        average=Config.SENTIMENT_AVERAGE_STRATEGY,
        zero_division=1,
    )
    recall = recall_score(
        ground_truth_labels,
        predicted_labels,
        average=Config.SENTIMENT_AVERAGE_STRATEGY,
        zero_division=1,
    )
    f1 = f1_score(
        ground_truth_labels,
        predicted_labels,
        average=Config.SENTIMENT_AVERAGE_STRATEGY,
        zero_division=1,
    )

    return accuracy, precision, recall, f1


def calculate_keyword_extraction_metrics(reference_keywords, extracted_keywords):
    flattened_reference_keywords = flatten_list(reference_keywords)
    flattened_extracted_keywords = flatten_list(extracted_keywords)

    if len(flattened_reference_keywords) != len(flattened_extracted_keywords):
        # Pad the shorter list with a placeholder value to align the lengths
        max_length = max(
            len(flattened_reference_keywords), len(flattened_extracted_keywords)
        )
        placeholder = "<PAD>"
        flattened_reference_keywords += [placeholder] * (
            max_length - len(flattened_reference_keywords)
        )
        flattened_extracted_keywords += [placeholder] * (
            max_length - len(flattened_extracted_keywords)
        )

    precision, recall, f1, _ = precision_recall_fscore_support(
        flattened_reference_keywords,
        flattened_extracted_keywords,
        average=Config.KEYWORD_EXTRACTION_AVERAGE_STRATEGY,
        zero_division=1,
    )

    return precision, recall, f1


def calculate_summary_generation_metrics(reference_summaries, generated_summaries):
    precision, recall, f1, _ = precision_recall_fscore_support(
        reference_summaries,
        generated_summaries,
        average=Config.SUMMARY_GENERATION_AVERAGE_STRATEGY,
        zero_division=1,
    )

    return precision, recall, f1
