from .keyword_extractor import KeywordExtractor
from .sentiment_classifier import SentimentClassifier
from .summary_generator import SummaryGenerator
import concurrent.futures
import logging
import time
from torch.utils.data import Dataset
import torch


class SectionDataset(Dataset):
    def __init__(self, sections):
        self.sections = sections

    def __len__(self):
        return len(self.sections)

    def __getitem__(self, idx):
        section = self.sections[idx]
        return section


class SectionProcessor:
    def __init__(self):
        self.sentiment_classifier = SentimentClassifier()
        self.keyword_extractor = KeywordExtractor()
        self.summary_generator = SummaryGenerator()
        self.logger = logging.getLogger(__name__)

    def process_sections(self, sections):
        analyzed_sections = []

        # Create the dataset
        dataset = SectionDataset(sections)

        # Create the data loader
        batch_size = 16
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            start_time = time.time()

            for batch in data_loader:
                sentiment_futures = [
                    executor.submit(
                        self.sentiment_classifier.classify_sentiment_transformer, sec
                    )
                    for sec in batch
                ]
                sentiment_spacy_futures = [
                    executor.submit(
                        self.sentiment_classifier.classify_sentiment_spacy, sec
                    )
                    for sec in batch
                ]
                summary_spacy_futures = [
                    executor.submit(self.summary_generator.generate_summary, sec)
                    for sec in batch
                ]
                keywords_rake_futures = [
                    executor.submit(self.keyword_extractor.extract_keywords_rake, sec)
                    for sec in batch
                ]
                keywords_textrank_futures = [
                    executor.submit(
                        self.keyword_extractor.extract_keywords_textrank, sec
                    )
                    for sec in batch
                ]
                keywords_spacy_futures = [
                    executor.submit(self.keyword_extractor.extract_keywords_spacy, sec)
                    for sec in batch
                ]
                keywords_huggingface_futures = [
                    executor.submit(
                        self.keyword_extractor.extract_keywords_huggingface, sec
                    )
                    for sec in batch
                ]
                keywords_yake_futures = [
                    executor.submit(self.keyword_extractor.extract_keywords_yake, sec)
                    for sec in batch
                ]

                for i, sec in enumerate(batch):
                    sentiment_transformer = sentiment_futures[i].result()
                    self.logger.info(
                        f"Sentiment Transformer task completed for section {i + 1}/{len(batch)}"
                    )
                    sentiment_spacy, sentiment_label_spacy = sentiment_spacy_futures[
                        i
                    ].result()
                    self.logger.info(
                        f"Sentiment Spacy task completed for section {i + 1}/{len(batch)}"
                    )
                    sentiment_spacy.pop("assessments", None)
                    sentiment_spacy.pop("ngrams", None)

                    summary_spacy = summary_spacy_futures[i].result()
                    self.logger.info(
                        f"Summary Spacy task completed for section {i + 1}/{len(batch)}"
                    )

                    keywords_rake = keywords_rake_futures[i].result()[:10]
                    self.logger.info(
                        f"Keywords Rake task completed for section {i + 1}/{len(batch)}"
                    )
                    keywords_textrank = keywords_textrank_futures[i].result()[:10]
                    self.logger.info(
                        f"Keywords Textrank task completed for section {i + 1}/{len(batch)}"
                    )
                    keywords_spacy = keywords_spacy_futures[i].result()[:10]
                    self.logger.info(
                        f"Keywords Spacy task completed for section {i + 1}/{len(batch)}"
                    )
                    keywords_huggingface = keywords_huggingface_futures[i].result()[:10]
                    self.logger.info(
                        f"Keywords Huggingface task completed for section {i + 1}/{len(batch)}"
                    )
                    keywords_yake = keywords_yake_futures[i].result()[:10]
                    self.logger.info(
                        f"Keywords Yake task completed for section {i + 1}/{len(batch)}"
                    )

                    analyzed_section = {
                        "sentiment_transformer": sentiment_transformer,
                        "sentiment_spacy": sentiment_spacy,
                        "sentiment_label_spacy": sentiment_label_spacy,
                        "summary_spacy": summary_spacy,
                        "keywords_rake": keywords_rake,
                        "keywords_textrank": keywords_textrank,
                        "keywords_spacy": keywords_spacy,
                        "keywords_huggingface": keywords_huggingface,
                        "keywords_yake": keywords_yake,
                    }
                    analyzed_sections.append(analyzed_section)

            end_time = time.time()
            overall_time = end_time - start_time
            self.logger.info(
                f"All processing tasks completed. Overall time taken: {overall_time} seconds"
            )

        return analyzed_sections
