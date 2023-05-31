# _news/scraper.py
import requests
from bs4 import BeautifulSoup
import logging
from langdetect import detect
import re


def scrape_and_process(url, headers, exclude_list):
    """
    Scrapes and processes the content of a given URL.
    Returns a dictionary containing the URL and processed article text, or None if an error occurs or the content is not valid.
    """

    def clean_text(text):
        """
        Cleans and preprocesses the given text.
        """
        if not text:
            return ""

        # Remove unwanted text
        text = re.sub(r"\[Website Name\]", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"This article is a paid publication and does not have journalistic editorial involvement of Hindustan Times.*$", "", text, flags=re.MULTILINE)

        # Clean and preprocess the text
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,?!]", "", text)  # Preserve periods, commas, question marks, and exclamation marks

        # Ensure text format
        text = text.replace("\n", "")

        return text

    try:
        with requests.Session() as session:
            r = session.get(url, headers=headers)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            results = soup.find_all("p")
            text = [res.text for res in results]
            article = " ".join(text)
            
            # Language filtering
            detected_lang = detect(article)
            if detected_lang != "en":
                return None
            
            # Clean and preprocess the article text
            article = clean_text(article)
            
            # Exclude articles containing specific phrases
            if any(exclude in article for exclude in exclude_list) or len(article) <= 50:
                return None
            
            return {"url": url, "article": article}
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making the request: {str(e)}")
        return None
        
    except AttributeError as e:
        logging.error(f"Error accessing attributes: {str(e)}")
        return None
        
    except Exception as e:
        logging.error(f"Error processing URL: {url}. Error: {str(e)}")
        return None
