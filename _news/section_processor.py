from concurrent.futures import ThreadPoolExecutor
import logging
import time
from torch.utils.data import Dataset
import torch
from config import Config
from _news.sentiment_classifier import SentimentClassifier
from _news.keyword_extractor import KeywordExtractor
from _news.summary_generator import SummaryGenerator


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

        # Adjust the batch size parameter to increase the batch size
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=Config.BATCH_SIZE, shuffle=True
        )

        with ThreadPoolExecutor(
            max_workers=Config.MAX_WORKERS
        ) as executor:  # Fine-tune the number of workers based on system capabilities
            start_time = time.time()
            for i, batch in enumerate(data_loader):
                sentiment_futures = [
                    executor.submit(
                        self.sentiment_classifier.classify_sentiment_transformer, sec
                    )
                    for sec in batch
                ]
                self.logger.info(
                    f"Sentiment Transformer task started for section {i + 1}/{len(batch)}"
                )
                sentiment_spacy_futures = [
                    executor.submit(
                        self.sentiment_classifier.classify_sentiment_spacy, sec
                    )
                    for sec in batch
                ]
                self.logger.info(
                    f"Sentiment Spacy task started for section {i + 1}/{len(batch)}"
                )
                summary_spacy_futures = [
                    executor.submit(self.summary_generator.generate_summary_spacy, sec)
                    for sec in batch
                ]
                self.logger.info(
                    f"Summary Spacy task started for section {i + 1}/{len(batch)}"
                )
                summary_transformer_futures = [
                    executor.submit(
                        self.summary_generator.generate_summary_transformer, sec
                    )
                    for sec in batch
                ]
                self.logger.info(
                    f"Summary Transformer task started for section {i + 1}/{len(batch)}"
                )
                keywords_rake_futures = [
                    executor.submit(self.keyword_extractor.extract_keywords_rake, sec)
                    for sec in batch
                ]
                self.logger.info(
                    f"Keyword Rake task started for section {i + 1}/{len(batch)}"
                )
                keywords_textrank_futures = [
                    executor.submit(
                        self.keyword_extractor.extract_keywords_textrank, sec
                    )
                    for sec in batch
                ]
                self.logger.info(
                    f"Keyword TextRank task started for section {i + 1}/{len(batch)}"
                )
                keywords_spacy_futures = [
                    executor.submit(self.keyword_extractor.extract_keywords_spacy, sec)
                    for sec in batch
                ]
                self.logger.info(
                    f"Keyword Spacy task started for section {i + 1}/{len(batch)}"
                )
                keywords_huggingface_futures = [
                    executor.submit(
                        self.keyword_extractor.extract_keywords_huggingface, sec
                    )
                    for sec in batch
                ]
                self.logger.info(
                    f"Keyword Transformer task started for section {i + 1}/{len(batch)}"
                )
                keywords_yake_futures = [
                    executor.submit(self.keyword_extractor.extract_keywords_yake, sec)
                    for sec in batch
                ]
                self.logger.info(
                    f"Keyword Yake task started for section {i + 1}/{len(batch)}"
                )

                for i, sec in enumerate(batch):
                    sentiment_transformer, sentiment_transformer_label = (
                        sentiment_futures[i].result().values()
                    )
                    self.logger.info(
                        f"Sentiment Transformer task completed for section {i + 1}/{len(batch)}"
                    )
                    sentiment_spacy, subjectivity_spacy = (
                        sentiment_spacy_futures[i].result().values()
                    )

                    self.logger.info(
                        f"Sentiment Spacy task completed for section {i + 1}/{len(batch)}"
                    )

                    summary_spacy = summary_spacy_futures[i].result()
                    self.logger.info(
                        f"Summary Spacy task completed for section {i + 1}/{len(batch)}"
                    )
                    summary_transformer = summary_transformer_futures[i].result()
                    self.logger.info(
                        f"Summary Transformer task completed for section {i + 1}/{len(batch)}"
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
                        "subjectivity_spacy": subjectivity_spacy,
                        "sentiment_label_transformer": sentiment_transformer_label,
                        "sentiment_label_spacy": self.sentiment_classifier.convert_polarity_to_label(
                            sentiment_spacy
                        ),
                        "summary_spacy": summary_spacy,
                        "summary_transformer": summary_transformer,
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
