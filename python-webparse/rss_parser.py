from keyword_search import launch_re
import feedparser

def rss_parser(RSS_FEEDS):

    rows = []
    for source, company, url in RSS_FEEDS:
        feed = feedparser.parse(url)
        print(source, "entries:", len(feed.entries), "status:", getattr(feed, "status", None), "bozo", feed.bozo)
        for e in feed.entries:
            title = e.get("title", "")
            description = e.get("description", "")
            content = e.get("content", "")
            published = e.get("published", e.get("updated", None))

            if launch_re.search(title) or launch_re.search(description) or launch_re.search(str(content)):
                rows.append({
                    "ticker": source,
                    "company_name": company,
                    "use_case": title,
                    "annoucement_date": published
                })
    print("RSS Parser Complete")
    return rows