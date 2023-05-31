# _news/url_cleaner.py
from urllib.parse import urlparse, urlunparse, unquote
import validators

def clean_urls(urls):
    """
    Cleans and validates the URLs in a given list
    """
    cleaned_urls = [
        unquote(
            urlunparse(
                (
                    urlparse(url).scheme.lower(), 
                    urlparse(url).netloc.lower().rstrip('/'), 
                    urlparse(url).path, 
                    '', 
                    '', 
                    ''
                )
            )
        )
        for url in urls 
        if validators.url(
            unquote(
                urlunparse(
                    (
                        urlparse(url).scheme.lower(), 
                        urlparse(url).netloc.lower().rstrip('/'), 
                        urlparse(url).path, 
                        '', 
                        '', 
                        ''
                    )
                )
            )
        )
    ]
    return cleaned_urls
