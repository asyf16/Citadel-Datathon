
import newspaper
from bs4 import BeautifulSoup
from keyword_search import launch_re

def news_parser(ticker, company, url):
    news_build = newspaper.build(url, memoize_articles = False) 
    rows = []

    for each_article in news_build.articles:
        try:
            each_article.download()
            each_article.parse()
        except:
            continue
        soup = BeautifulSoup(each_article.html, 'html.parser')
        published_date = soup.find(class_='_amum') or soup.find(class_='_8w6f _8wl0 _8w6h') or each_article.publish_date
        title = each_article.title  
        content = each_article.text
        
        if (title and published_date):

            if launch_re.search(title) or launch_re.search(content):
                rows.append({
                    "ticker": ticker,
                    "company_name": company,
                    "use_case": title,
                    "annoucement_date": published_date
                })
    print("News Parser Complete")
    return rows