import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from _news.section_processor import SectionProcessor
from config import Config
import time
from memory_profiler import profile


@profile
def ml_using_trnf(articles):
    logging.basicConfig(level=Config.LOG_LEVEL)
    processor = SectionProcessor()
    analyzed_articles = {}

    # Create an executor for IO-bound tasks
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        # Create an executor for CPU-bound tasks
        with ProcessPoolExecutor() as cpu_executor:
            article_tasks = []
            for i, article in enumerate(articles):
                url = article.get("url", "")
                text = article.get("article", "")
                logging.info(f"Starting processing for article {i + 1}/{len(articles)}")

                # Create a unique key for the current article
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f"_{i}"

                # Create a dictionary to store the analyzed data for the current article
                analyzed_data = {
                    "text": text,  # Store the full article text
                    "url": url,  # Store the URL
                }

                # Submit the processing tasks to the executors
                sentiment_transformer_task = executor.submit(
                    processor.sentiment_classifier.classify_sentiment_transformer, text
                )
                logging.info(
                    f"Submitted Sentiment Transformer task for article {i + 1}/{len(articles)}"
                )
                # sentiment_spacy_task = cpu_executor.submit(processor.sentiment_classifier.classify_sentiment_spacy, text)
                # logging.info(f"Submitted Sentiment Spacy task for article {i + 1}/{len(articles)}")
                # summary_spacy_task = cpu_executor.submit(processor.summary_generator.generate_spacy, text)
                # logging.info(f"Submitted Summary Spacy task for article {i + 1}/{len(articles)}")
                summary_transformer_task = executor.submit(
                    processor.summary_generator.generate_summary, text
                )
                logging.info(
                    f"Submitted Summary Transformer task for article {i + 1}/{len(articles)}"
                )
                # keywords_rake_task = cpu_executor.submit(processor.keyword_extractor.extract_keywords_rake, text)
                # logging.info(f"Submitted Keywords RAKE task for article {i + 1}/{len(articles)}")
                # keywords_textrank_task = cpu_executor.submit(processor.keyword_extractor.extract_keywords_textrank, text)
                # logging.info(f"Submitted Keywords TextRank task for article {i + 1}/{len(articles)}")
                # keywords_spacy_task = cpu_executor.submit(processor.keyword_extractor.extract_keywords_spacy, text)
                # logging.info(f"Submitted Keywords Spacy task for article {i + 1}/{len(articles)}")
                # keywords_huggingface_task = executor.submit(processor.keyword_extractor.extract_keywords_huggingface, text)
                # logging.info(f"Submitted Keywords HuggingFace task for article {i + 1}/{len(articles)}")
                keywords_yake_task = cpu_executor.submit(
                    processor.keyword_extractor.extract_keywords_yake, text
                )
                logging.info(
                    f"Submitted Keywords YAKE task for article {i + 1}/{len(articles)}"
                )

                # Store the tasks for future retrieval
                article_tasks.append(
                    (
                        timestamp,
                        analyzed_data,
                        sentiment_transformer_task,
                        # sentiment_spacy_task,
                        # summary_spacy_task,
                        summary_transformer_task,
                        # keywords_rake_task,
                        # keywords_textrank_task,
                        # keywords_spacy_task,
                        # keywords_huggingface_task,
                        keywords_yake_task,
                    )
                )

            # Process the completed tasks and populate the analyzed_articles dictionary
            for (
                timestamp,
                analyzed_data,
                sentiment_transformer_task,
                # sentiment_spacy_task,
                # summary_spacy_task,
                summary_transformer_task,
                # keywords_rake_task,
                # keywords_textrank_task,
                # keywords_spacy_task,
                # keywords_huggingface_task,
                keywords_yake_task,
            ) in article_tasks:
                sentiment_transformer = sentiment_transformer_task.result()
                logging.info(
                    f"Sentiment Transformer task completed for article {i + 1}/{len(articles)}: {sentiment_transformer_task.done()}"
                )
                # sentiment_spacy = sentiment_spacy_task.result()
                # logging.info(f"Sentiment Spacy task completed for article {i + 1}/{len(articles)}: {sentiment_spacy_task.done()}")
                sentiment_label_transformer = sentiment_transformer["overall_label"]
                # sentiment_label_spacy = processor.sentiment_classifier.convert_polarity_to_label(sentiment_spacy['polarity'])
                # analyzed_data['sentiment_transformer'] = sentiment_transformer['overall_score']
                # analyzed_data['sentiment_spacy'] = sentiment_spacy
                # analyzed_data['sentiment_label_spacy'] = sentiment_label_spacy
                analyzed_data[
                    "sentiment_label_transformer"
                ] = sentiment_label_transformer

                # summary_spacy = summary_spacy_task.result()
                # logging.info(f"Summary Transformer task completed for article {i + 1}/{len(articles)}: {summary_spacy_task.done()}")
                summary_transformer = summary_transformer_task.result()
                logging.info(
                    f"Summary Transformer task completed for article {i + 1}/{len(articles)}: {summary_transformer_task.done()}"
                )
                # analyzed_data['summary_spacy'] = summary_spacy
                analyzed_data["summary_transformer"] = summary_transformer

                # keywords_rake = keywords_rake_task.result()[:10]
                # logging.info(f"Keyword rake task completed for article {i + 1}/{len(articles)}: {keywords_rake_task.done()}")
                # keywords_textrank = keywords_textrank_task.result()[:10]
                # logging.info(f"Keyword textrank task completed for article {i + 1}/{len(articles)}: {keywords_textrank_task.done()}")
                # keywords_spacy = keywords_spacy_task.result()[:10]
                # logging.info(f"Keyword spacy task completed for article {i + 1}/{len(articles)}: {keywords_spacy_task.done()}")
                # keywords_huggingface = keywords_huggingface_task.result()[:10]
                # logging.info(f"Keyword Transformer task completed for article {i + 1}/{len(articles)}: {keywords_huggingface_task.done()}")
                keywords_yake = keywords_yake_task.result()[:10]
                logging.info(
                    f"Keyword Yake task completed for article {i + 1}/{len(articles)}: {keywords_yake_task.done()}"
                )
                # analyzed_data['keywords_rake'] = keywords_rake
                # analyzed_data['keywords_textrank'] = keywords_textrank
                # analyzed_data['keywords_spacy'] = keywords_spacy
                # analyzed_data['keywords_huggingface'] = keywords_huggingface
                analyzed_data["keywords_yake"] = keywords_yake

                analyzed_articles[timestamp] = analyzed_data

    logging.info("Finished processing all articles.")
    return analyzed_articles
