import aiohttp
import logging
from bs4 import BeautifulSoup
import re
from langdetect import detect
from config import Config

logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("analysis.log"), logging.StreamHandler()],
)


async def scrape_and_process(url, headers, exclude_list):
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
        text = re.sub(
            r"This article is a paid publication and does not have journalistic editorial involvement of Hindustan Times.*$",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Clean and preprocess the text
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(
            r"[^\w\s.,?!]", "", text
        )  # Preserve periods, commas, question marks, and exclamation marks

        # Ensure text format
        text = text.replace("\n", "")

        return text

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }

            async with session.get(url, headers=headers) as r:
                r.raise_for_status()
                soup = BeautifulSoup(await r.text(), "lxml")
                article = ""

                # Find article text in different HTML structures
                article_tags = soup.find_all(["p", "div", "article", "section"])
                for tag in article_tags:
                    text = tag.get_text(separator=" ")
                    if len(text) > len(article):
                        article = text

                # Language filtering
                detected_lang = detect(article)
                if detected_lang != "en":
                    logging.warning(
                        f"Detected language is {detected_lang}, skipping URL: {url}"
                    )
                    return None

                # Clean and preprocess the article text
                article = clean_text(article)

                # Exclude articles containing specific phrases
                if (
                    any(
                        re.search(rf"\b{re.escape(exclude)}\b", article)
                        for exclude in exclude_list
                    )
                    or len(article) <= 50
                ):
                    excluded_phrases = [
                        exclude
                        for exclude in exclude_list
                        if re.search(rf"\b{re.escape(exclude)}\b", article)
                    ]
                    logging.warning(
                        f"Excluded URL due to specific phrases or short length: {url}"
                    )
                    logging.debug(f"Article text: {article}")
                    logging.debug(f"Article length: {len(article)}")
                    logging.debug(f"Excluded phrases: {excluded_phrases}")
                    return None

                logging.info(f"Scraping and processing of URL successful: {url}")
                return {"url": url, "article": article}

    except aiohttp.ClientError as e:
        logging.error(f"Error making the request: {str(e)}")
        return None

    except AttributeError as e:
        logging.error(f"Error accessing attributes: {str(e)}")
        return None

    except Exception as e:
        logging.error(f"Error processing URL: {url}. Error: {str(e)}")
        return None
