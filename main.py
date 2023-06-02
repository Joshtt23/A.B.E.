import tkinter as tk
from tkinter import ttk
import json
import time
import logging
import os
from config import Config
from _news.news_fetcher import fetch_news
from _news.url_cleaner import clean_urls
from _news.scraper import scrape_and_process
from _news.analyzer import ml_using_trnf
from _news.score_calculator import (
    calculate_sentiment_metrics,
    calculate_keyword_extraction_metrics,
    calculate_summary_generation_metrics,
)
import asyncio
from SAMPLE import SAMPLE
import traceback

EXCLUDE_LIST = Config.EXCLUDE_LIST

HEADERS = {
    "User-Agent": Config.USER_AGENT,
    "Accept-Language": Config.ACCEPT_LANGUAGE,
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("analysis.log"),  # Log messages to a file
        logging.StreamHandler(),  # Log messages to the console
    ],
)
logger = logging.getLogger(__name__)


class LoadingBar(tk.Tk):
    def __init__(self, total_stages, icon_path):
        super().__init__()
        self.title("Live Analysis Progress")
        self.set_window_icon(icon_path)

        self.progress_bar = ttk.Progressbar(
            self, length=300, mode="determinate", maximum=100
        )
        self.progress_bar.pack(pady=10)
        self.progress_label = tk.Label(self, text="Initializing...")
        self.progress_label.pack(pady=5)

        self.cancelled = False
        self.total_stages = total_stages
        self.current_stage = 0

    def update_progress(self, progress, message):
        self.progress_bar["value"] = progress
        self.progress_label.config(text=message)
        self.update()

    def increment_current_stage(self):
        self.current_stage += 1
        progress = (self.current_stage / self.total_stages) * 100
        self.update_progress(
            progress, f"Processing article {self.current_stage}/{self.total_stages}"
        )

    def update_total_stages(self, total_stages):
        self.total_stages = total_stages

    def cancel_analysis(self):
        self.cancelled = True
        self.destroy()

    def set_window_icon(self, icon_path):
        icon_path = os.path.join(os.path.dirname(__file__), icon_path)
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)


async def perform_live_analysis(loading_bar=None):
    try:
        # Fetch news articles
        logger.info("Fetching news articles...")
        # raw_urls = await fetch_news()  # Await fetch_news() function
        if loading_bar and loading_bar.cancelled:
            return
        # logger.info(f"Fetched {len(raw_urls)} news articles.")

        if loading_bar:
            loading_bar.update_progress(0, "Cleaning and validating URLs...")

        # Clean and validate URLs
        logger.info("Cleaning and validating URLs...")
        # cleaned_urls = await clean_urls(raw_urls)  # Await clean_urls() function
        if loading_bar and loading_bar.cancelled:
            return
        # logger.info(f"Cleaned and validated {len(cleaned_urls)} URLs.")

        # if loading_bar:
        # Update total stages in loading bar
        # loading_bar.update_total_stages(len(cleaned_urls))

        # # Scrape and process articles
        # articles = []
        # for i, url in enumerate(cleaned_urls):
        #     if loading_bar and loading_bar.cancelled:
        #         return
        #     if loading_bar:
        #         loading_bar.increment_current_stage()
        #     logger.info(f"Processing article {i + 1}/{len(cleaned_urls)}")
        #     article = await scrape_and_process(
        #         url, headers=HEADERS, exclude_list=EXCLUDE_LIST
        #     )  # Await scrape_and_process() function
        #     if article is not None:
        #         articles.append(article)

        articles = SAMPLE

        if loading_bar:
            loading_bar.update_progress(100, "Performing analysis on articles...")
        logger.info("Performing analysis on articles...")
        analyzed_articles = ml_using_trnf(articles)
        if loading_bar and loading_bar.cancelled:
            return
        logger.info("Analysis completed.")

        # Calculate sentiment metrics
        logger.info("Calculating sentiment metrics...")
        try:
            ground_truth_sentiment = [
                article["sentiment_spacy"]["polarity"]
                for article in analyzed_articles.values()
            ]
            predicted_sentiment = [
                article["sentiment_transformer"]
                for article in analyzed_articles.values()
            ]

            threshold = 0.2
            sentiment_metrics = calculate_sentiment_metrics(
                ground_truth_sentiment, predicted_sentiment, threshold
            )
            logger.info("Sentiment metrics calculated.")
        except KeyError:
            logger.error("Unable to calculate sentiment analysis metrics")
            sentiment_metrics = None

        # Calculate keyword extraction metrics
        logger.info("Calculating keyword extraction metrics...")
        try:
            reference_keywords = [
                article["keywords_spacy"] for article in analyzed_articles.values()
            ]
            extracted_keywords = [
                article["keywords_rake"] for article in analyzed_articles.values()
            ]
            keyword_extraction_metrics = calculate_keyword_extraction_metrics(
                reference_keywords, extracted_keywords
            )
            logger.info("Keyword extraction metrics calculated.")
        except KeyError:
            logger.error("Unable to calculate keyword extraction metrics.")
            keyword_extraction_metrics = None

        # Calculate summary generation metrics
        logger.info("Calculating summary generation metrics...")
        try:
            reference_summaries = [
                article["summary_spacy"] for article in analyzed_articles.values()
            ]
            generated_summaries = [
                article["summary_transformer"] for article in analyzed_articles.values()
            ]
            summary_generation_metrics = calculate_summary_generation_metrics(
                reference_summaries, generated_summaries
            )
            logger.info("Summary generation metrics calculated.")
        except KeyError:
            logger.error("Unable to calculate summary generation metrics.")
            summary_generation_metrics = None

        # Store metrics in a dictionary
        metrics = {
            "sentiment_metrics": sentiment_metrics,
            "keyword_extraction_metrics": keyword_extraction_metrics,
            "summary_generation_metrics": summary_generation_metrics,
        }

        # Write the analyzed articles and metrics to result.json
        result = {"analyzed_articles": analyzed_articles, "metrics": metrics}

        try:
            script_directory = os.path.dirname(os.path.abspath(__file__))
            result_file_path = os.path.join(script_directory, "result.json")

            with open(result_file_path, "w") as f:
                json.dump(result, f)

            logger.info("Result written to result.json")
        except Exception:
            logger.error("Error occurred while writing result to result.json")
            traceback.print_exc()
        if loading_bar:
            loading_bar.update_progress(100, "Analysis complete.")

        logger.info("Analysis complete.")

        if loading_bar:
            loading_bar.destroy()  # Close the window after analysis completes successfully

    except tk.TclError:
        logger.info("Analysis paused. Click 'Resume' to continue.")


def run_with_gui(loading_bar):
    loading_bar.update_progress(0, "Initializing...")
    loading_bar.cancelled = False

    def on_cancel():
        loading_bar.cancel_analysis()

    def on_resume():
        loading_bar.resume_analysis()

    cancel_button = tk.Button(loading_bar, text="Cancel", width=10, command=on_cancel)
    cancel_button.pack(pady=5)

    resume_button = tk.Button(loading_bar, text="Resume", width=10, command=on_resume)
    resume_button.pack(pady=5)
    resume_button.pack_forget()

    loading_bar.protocol("WM_DELETE_WINDOW", on_cancel)
    loading_bar.resizable(False, False)
    loading_bar.update_progress(0, "Fetching news articles...")
    asyncio.run(
        perform_live_analysis(loading_bar)
    )  # Run the async function with asyncio.run()

    loading_bar.mainloop()


if __name__ == "__main__":
    loading_bar = LoadingBar(total_stages=0, icon_path="favicon.ico")
    run_with_gui(loading_bar)
