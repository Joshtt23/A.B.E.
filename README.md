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
2. Run News ML.bat as an administrator.
   - The installation script will automatically create a virtual environment (venv) and install all the required dependencies.

## Usage

To use the News ML application, follow these steps:

1. Run News ML.bat as an administrator.
   - If prompted, grant administrator privileges to the script.
   - The application will start automatically once the administrator check is successful.
2. Complete tasks using the user interface:
   - The application will fetch news articles, process them, perform analysis, and calculate metrics.
   - The analyzed articles and metrics will be saved in `result.json`.
   - Note: It is important to run News ML.bat as an administrator to ensure the application functions correctly.

## Files

The News ML application consists of the following files and folders:

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
- `tnt/`: Folder containing test and train scripts.
  - `ner/`: Folder for named entity recognition (NER) tasks.
    - `test.py`: Script for running NER tests.
    - `train.py`: Script for training NER models.
  - `sentiment_analysis/`: Folder for sentiment analysis tasks.
    - `test.py`: Script for running sentiment analysis tests.
    - `train.py`: Script for training sentiment analysis models.
  - `summary_generation/`: Folder for summary generation tasks.
    - `test.py`: Script for running summary generation tests.
    - `train.py`: Script for training summary generation models.
- `run.py`: Script to run the application's user interface (UI).
- `News ML.bat`: Batch file to start the application automatically.
- `result.json`: JSON file to store the analyzed articles and metrics.
- `README.md`: This file.

## Contributing

Contributions to the News ML project are welcome! If you have any ideas, improvements, or bug fixes, please open an issue or submit a pull request.

## License

The News ML project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
