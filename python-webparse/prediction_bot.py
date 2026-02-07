import re
import feedparser
from threading import Thread, Lock
from time import sleep
from datetime import datetime, timedelta
import pandas as pd

RSS_FEEDS = [
    ("GOOGL", "%a, %d %b %Y %H:%M:%S %z", "https://deepmind.google/blog/rss.xml"),
    ("META", "%a, %d %b %Y %H:%M:%S %z", "https://engineering.fb.com/feed/"),
    ("META", "%a, %d %b %Y %H:%M:%S", "https://research.facebook.com/feed/"), 
    ("NVDA", "%Y-%m-%dT%H:%M:%SZ", "https://developer.nvidia.com/blog/feed"),
    ("MSFT", "%a, %d %b %Y %H:%M:%S", "https://www.microsoft.com/en-us/ai/blog/feed/"),
    ("AMZN", "%a, %d %b %Y %H:%M:%S %z", "https://aws.amazon.com/blogs/machine-learning/feed/")
]

LAUNCH_PATTERNS = [
    r"\blaunch\b", r"\bintroducing\b", r"\bannounce\b", r"\brelease\b",
    r"\bnow available\b", r"\blaunching\b"
]
launch_re = re.compile("|".join(LAUNCH_PATTERNS), re.IGNORECASE)

class ScrapeRSS(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.mutex = Lock()
        self._quit = False

    def stopped(self):
        self.mutex.acquire()
        val = self._quit
        self.mutex.release()
        return val

    def stop(self):
        self.mutex.acquire()
        self._quit = True
        self.mutex.release()

    def run(self):
        while True:
            if self.stopped():
                return
            print("Starting Web Scraping Cycle")
            score_df = pd.read_csv("stock_scores.csv")
            for ticker, time_format, url in RSS_FEEDS:
                feed = feedparser.parse(url)
                for e in feed.entries:
                    published_date = e.get("published", e.get("updated", None))
                    try:
                        published_date = datetime.strptime(published_date, time_format)
                    except:
                        continue
                    naive_published_date = published_date.replace(tzinfo=None)
                    three_days_ago = datetime.now() - timedelta(days=3)
                    if naive_published_date < three_days_ago:
                        continue
                    title = e.get("title", "")
                    description = e.get("description", "")
                    content = e.get("content", "")
                    if launch_re.search(title) or launch_re.search(description) or launch_re.search(str(content)):
                        print("Article found for", ticker)
                        print("Article title:", title)
                        company_data = score_df[score_df["Ticker"] == ticker].head(5)
                        company_data = company_data.drop('Ticker', axis=1)
                        print("Top 5 affected firms:", company_data)
            print("Web Scraping Cycle Completed")
            sleep(1200)

if __name__ in {"__main__", "__mp_main__"}:
    RSS_Scraper = ScrapeRSS()
    RSS_Scraper.start()
