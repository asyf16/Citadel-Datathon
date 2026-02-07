import pandas as pd
from rss_parser import rss_parser
from xml_parser import xml_parser
from news_parser import news_parser

RSS_FEEDS = [
    ("GOOGL", "Google", "https://deepmind.google/blog/rss.xml"),
    ("META", "Meta", "https://engineering.fb.com/feed/"),
    ("META", "Meta", "https://research.facebook.com/feed/"), 
    ("NVDA", "Nvidia", "https://developer.nvidia.com/blog/feed"),
    ("MSFT", "Microsoft", "https://www.microsoft.com/en-us/ai/blog/feed/"),
    ("AMZN", "Amazon", "https://aws.amazon.com/blogs/machine-learning/feed/")
]
NETFLIX_URL = "https://netflixtechblog.com/feed?topic=artificial-intelligence"

NEWSPARSER_URLS = [
    ("META", "Meta", "https://ai.meta.com/blog/"),
    ("NVDA", "Nvidia", "https://www.nvidia.com/en-us/research/news/"),
    ("MSFT", "Microsoft", "https://www.microsoft.com/en-us/ai/blog/"),
    ("AMZN", "Amazon", "https://www.aboutamazon.com/artificial-intelligence-ai-news")
]

if __name__ == '__main__':
    data = []
    xml_rows = xml_parser(NETFLIX_URL)
    rss_rows = rss_parser(RSS_FEEDS)
    data += xml_rows
    data += rss_rows

    for ticker, company, url in NEWSPARSER_URLS:
        company_data = news_parser(ticker, company, url)
        data += company_data

    parsed_df = pd.DataFrame(data)

    parsed_df = pd.DataFrame(parsed_df).drop_duplicates(subset=["use_case"])
    og_df = pd.read_csv('enterprise_ai_adoption_internet_events.csv')

    print(len(og_df))

    result = pd.concat([og_df, parsed_df], ignore_index=True).fillna('')
    result.to_csv("final_model_data.csv", index=False)



