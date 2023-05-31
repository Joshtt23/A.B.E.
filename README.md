# News ML

News ML is a Python application that fetches news articles, performs analysis on the articles, and calculates various metrics such as sentiment, keyword extraction, and summary generation. It utilizes machine learning models for sentiment analysis, keyword extraction, and text summarization.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Contributing](#contributing)
- [License](#license)

## Overview

The News ML application fetches news articles based on a specified search term using the Microsoft Bing News Search API. It then cleans and validates the URLs, scrapes and processes the articles' content, performs analysis using machine learning models, and calculates various metrics.

The application consists of the following components:

- `news_fetcher.py`: Fetches news articles using the Bing News Search API.
- `url_cleaner.py`: Cleans and validates URLs.
- `scraper.py`: Scrapes and processes the content of news articles.
- `analyzer.py`: Performs analysis on the articles, including sentiment analysis, keyword extraction, and summary generation.
- `score_calculator.py`: Calculates metrics such as sentiment metrics, keyword extraction metrics, and summary generation metrics.
- `section_processor.py`: Processes individual sections of an article, including sentiment analysis, keyword extraction, and summary generation.
- `sentiment_classifier.py`: Classifies the sentiment of text using machine learning models.
- `keyword_extractor.py`: Extracts keywords from text using various algorithms.
- `summary_generator.py`: Generates summaries of text using machine learning models.

## Installation

To install and run the News ML application, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/news-ml.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage

To use the News ML application, you need to provide your API key for the Microsoft Bing News Search API. Replace `'YOUR_API_KEY'` in `config.py` with your actual API key.

After configuring the API key, you can run the application by executing `main.py`.

`python main.py`

The application will fetch news articles, process them, perform analysis, and calculate metrics. The analyzed articles and metrics will be saved in `result.json`.

## Files

The News ML application consists of the following files:

- `main.py`: The entry point of the application.
- `config.py`: Configuration settings for the application.
- `news_fetcher.py`: Fetches news articles using the Microsoft Bing News Search API.
- `url_cleaner.py`: Cleans and validates URLs.
- `scraper.py`: Scrapes and processes the content of news articles.
- `analyzer.py`: Performs analysis on the articles, including sentiment analysis, keyword extraction, and summary generation.
- `score_calculator.py`: Calculates metrics such as sentiment metrics, keyword extraction metrics, and summary generation metrics.
- `section_processor.py`: Processes individual sections of an article, including sentiment analysis, keyword extraction, and summary generation.
- `sentiment_classifier.py`: Classifies the sentiment of text using machine learning models.
- `keyword_extractor.py`: Extracts keywords from text using various algorithms.
- `summary_generator.py`: Generates summaries of text using machine learning models.
- `README.md`: This file.

## Contributing

Contributions to the News ML project are welcome! If you have any ideas, improvements, or bug fixes, please open an issue or submit a pull request.

## License

The News ML project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
