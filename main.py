import json
import time
import logging
from _news.news_fetcher import fetch_news
from _news.url_cleaner import clean_urls
from _news.scraper import scrape_and_process
from _news.analyzer import ml_using_trnf
from _news.score_calculator import calculate_sentiment_metrics, calculate_keyword_extraction_metrics, calculate_summary_generation_metrics
from config import Config

EXCLUDE_LIST = Config.EXCLUDE_LIST

HEADERS = {
    "User-Agent": Config.USER_AGENT,
    "Accept-Language": Config.ACCEPT_LANGUAGE,
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Fetch news articles
    logger.info("Fetching news articles...")
    start_time = time.time()
    raw_urls = fetch_news()
    logger.info(f"Fetched {len(raw_urls)} news articles.")

    # Clean and validate URLs
    logger.info("Cleaning and validating URLs...")
    cleaned_urls = clean_urls(raw_urls)
    logger.info(f"Cleaned and validated {len(cleaned_urls)} URLs.")

    # Scrape and process articles
    logger.info("Scraping and processing articles...")
    articles = []

    for i, url in enumerate(cleaned_urls):
        logger.info(f"Processing article {i+1}/{len(cleaned_urls)}")
        article = scrape_and_process(url, headers=HEADERS, exclude_list=EXCLUDE_LIST)
        if article is not None:
            articles.append(article)

    logger.info(f"Scraped and processed {len(articles)} articles.")
    elapsed_time = time.time() - start_time
    logger.info(f"Scraping and processing completed in {elapsed_time:.2f} seconds.")

    # Perform analysis on articles
    logger.info("Performing analysis on articles...")
    start_time = time.time()
    analyzed_articles = ml_using_trnf(articles)
    logger.info("Analysis completed.")
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds.")

    # Calculate sentiment metrics
    logger.info("Calculating sentiment metrics...")
    start_time = time.time()
    ground_truth_sentiment = [article['sentiment_spacy']['polarity'] for article in analyzed_articles.values()]
    predicted_sentiment = [article['sentiment_transformer'] for article in analyzed_articles.values()]

    # Convert sentiment scores to categories
    threshold = 0.2  # Adjust the threshold as needed
    ground_truth_categories = ['Positive' if s > threshold else 'Negative' if s < -threshold else 'Neutral' for s in ground_truth_sentiment]
    predicted_categories = ['Positive' if s > threshold else 'Negative' if s < -threshold else 'Neutral' for s in predicted_sentiment]

    sentiment_metrics = calculate_sentiment_metrics(ground_truth_categories, predicted_categories)
    logger.info("Sentiment metrics calculated.")
    elapsed_time = time.time() - start_time
    logger.info(f"Sentiment metrics calculated in {elapsed_time:.2f} seconds.")

    # Calculate keyword extraction metrics
    logger.info("Calculating keyword extraction metrics...")
    start_time = time.time()

    try:
        reference_keywords = [article['keywords_spacy'] for article in analyzed_articles.values()]
        extracted_keywords = [article['keywords_rake'] for article in analyzed_articles.values()]
        keyword_extraction_metrics = calculate_keyword_extraction_metrics(reference_keywords, extracted_keywords)
        logger.info("Keyword extraction metrics calculated.")
    except KeyError:
        logger.error("Unable to calculate keyword extraction metrics. Key 'keywords_spacy' or 'keywords_rake' not found in analyzed articles.")
        keyword_extraction_metrics = None

    elapsed_time = time.time() - start_time
    logger.info(f"Keyword extraction metrics calculated in {elapsed_time:.2f} seconds.")

    # Calculate summary generation metrics
    logger.info("Calculating summary generation metrics...")
    start_time = time.time()

    try:
        reference_summaries = [article['summary_spacy'] for article in analyzed_articles.values()]
        generated_summaries = [article['summary_transformer'] for article in analyzed_articles.values()]
        summary_generation_metrics = calculate_summary_generation_metrics(reference_summaries, generated_summaries)
        logger.info("Summary generation metrics calculated.")
    except KeyError:
        logger.error("Unable to calculate summary generation metrics. Key 'summary_spacy' or 'summary_transformer' not found in analyzed articles.")
        summary_generation_metrics = None

    elapsed_time = time.time() - start_time
    logger.info(f"Summary generation metrics calculated in {elapsed_time:.2f} seconds.")

    # Store metrics in a dictionary
    metrics = {
        'sentiment_metrics': sentiment_metrics,
        'keyword_extraction_metrics': keyword_extraction_metrics,
        'summary_generation_metrics': summary_generation_metrics
    }

    # Write the analyzed articles and metrics to result.json
    result = {
        'analyzed_articles': analyzed_articles,
        'metrics': metrics
    }

    with open("result.json", "w") as f:
        json.dump(result, f)
        logger.info("Result saved to result.json.")
