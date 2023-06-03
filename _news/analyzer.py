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

                # Split the article into sections
                sections = text.split("\n\n")

                # Create a unique key for the current article
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f"_{i}"

                # Create a dictionary to store the analyzed data for the current article
                analyzed_data = {
                    "text": text,  # Store the full article text
                    "url": url,  # Store the URL
                }

                # Process the sections using the SectionProcessor
                analyzed_sections = processor.process_sections(sections)

                # Store the analyzed sections in the analyzed data
                analyzed_data["sections"] = analyzed_sections

                # Store the data for future retrieval
                analyzed_articles[timestamp] = analyzed_data
                logging.info(f"Finished processing for article {i + 1}/{len(articles)}")

    logging.info("Finished processing all articles.")
    return analyzed_articles
