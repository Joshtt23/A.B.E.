import datetime
from config import Config
import requests
import logging

def fetch_news():
    """
    Fetches the news based on a given search term from Microsoft Bing News Search API
    """
    url = 'https://api.bing.microsoft.com/v7.0/news/search'

    headers = {"Ocp-Apim-Subscription-Key": Config.API_KEY}
    params = {
    "q": Config.SEARCH_TERM,
    "count": Config.NEWS_COUNT,
    "freshness": "Day",
    "textFormat": "Raw",
    "category": "Business",  # Set category to business or finance
    "sortBy": "Date",  # Sort articles by date
    "mkt": "en-US",  # Set market to English (United States)
    "since": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),  # Set the "since" parameter to retrieve articles posted in the last 24 hours
}

    try:
        with requests.Session() as session:
            response = session.get(url, headers=headers, params=params)
            response.raise_for_status()
            news = response.json()

            urls = [article["url"] for article in news["value"]]
            logging.info("News fetch successful.")
            return urls
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {str(e)}, Status code: {response.status_code}")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making the request: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error retrieving news: {str(e)}")
        return []