import re, feedparser
from threading import Thread
from datetime import datetime, timedelta
import pandas as pd
import tkinter as tk
from tkinter import ttk

RSS_FEEDS = [
    ("GOOGL", "%a, %d %b %Y %H:%M:%S %z", "https://deepmind.google/blog/rss.xml"),
    ("META",  "%a, %d %b %Y %H:%M:%S %z", "https://engineering.fb.com/feed/"),
    ("META",  "%a, %d %b %Y %H:%M:%S",    "https://research.facebook.com/feed/"),
    ("NVDA",  "%Y-%m-%dT%H:%M:%SZ",       "https://developer.nvidia.com/blog/feed"),
    ("MSFT",  "%a, %d %b %Y %H:%M:%S",    "https://www.microsoft.com/en-us/ai/blog/feed/"),
    ("AMZN",  "%a, %d %b %Y %H:%M:%S %z", "https://aws.amazon.com/blogs/machine-learning/feed/"),
]

LAUNCH_PATTERNS = [r"\blaunch\b", r"\bintroducing\b", r"\bannounce\b", r"\brelease\b", r"\bnow available\b", r"\blaunching\b"]
launch_re = re.compile("|".join(LAUNCH_PATTERNS), re.IGNORECASE)

def parse_dt(s, fmt):
    if not s: return None
    try: return datetime.strptime(s, fmt).replace(tzinfo=None)
    except: return None

def main():
    score_df = pd.read_csv("stock_scores.csv")
    q = [] 
    running = {"v": False}

    root = tk.Tk()
    root.title("AI Announcements News Predictive Bot")
    root.geometry("820x520")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TButton", padding=10)
    style.configure("Title.TLabel", font=("SF Pro Display", 16, "bold"))
    style.configure("Sub.TLabel", font=("SF Pro Text", 11))

    top = ttk.Frame(root, padding=16)
    top.pack(fill="x")
    ttk.Label(top, text="AI Announcements News Predictive Bot", style="Title.TLabel").pack(anchor="w")

    mid = ttk.Frame(root, padding=(16, 0, 16, 16))
    mid.pack(fill="both", expand=True)

    toolbar = ttk.Frame(mid)
    toolbar.pack(fill="x", pady=(12, 10))
    btn = ttk.Button(toolbar, text="Run scan (last 3 days)")
    btn.pack(side="left")

    out_frame = ttk.Frame(mid)
    out_frame.pack(fill="both", expand=True)

    text = tk.Text(out_frame, wrap="word", bd=0, padx=12, pady=10, font=("SF Pro Text", 11))
    text.pack(side="left", fill="both", expand=True)
    sb = ttk.Scrollbar(out_frame, command=text.yview)
    sb.pack(side="right", fill="y")
    text.configure(yscrollcommand=sb.set)

    def log(msg):
        text.insert("end", msg + "\n")
        text.see("end")

    def worker():
        running["v"] = True
        three_days_ago = datetime.now() - timedelta(days=3)

        hits = 0
        for ticker, time_format, url in RSS_FEEDS:
            feed = feedparser.parse(url)
            for e in getattr(feed, "entries", []):
                dt = parse_dt(e.get("published") or e.get("updated"), time_format)
                if not dt or dt < three_days_ago: 
                    continue

                title = e.get("title", "")
                desc = e.get("description", "")
                content = str(e.get("content", ""))

                if launch_re.search(title) or launch_re.search(desc) or launch_re.search(content):
                    company_data = score_df[score_df["Ticker"] == ticker].head(5)
                    company_data = company_data.drop('Ticker', axis=1)
                    hits += 1
                    q.append(("log", f"❗ Announcement made by {ticker} | {dt.strftime('%Y-%m-%d %H:%M')}"))
                    q.append(("log", f"Article title: {title.strip()}"))
                    q.append(("log", "📈 Top 5 stocks by score:\n" + company_data.to_string(index=False)))
                    q.append(("log", ""))

        running["v"] = False
        q.append(("enable", True))

    def pump():
        while q:
            kind, val = q.pop(0)
            if kind == "log": log(val)
            elif kind == "enable": btn.config(state=("normal" if val else "disabled"))
        root.after(80, pump)

    def start():
        if running["v"]: 
            return
        btn.config(state="disabled")
        log("Starting Web Scraping Cycle\n")
        Thread(target=worker, daemon=True).start()

    btn.config(command=start)
    pump()
    root.mainloop()

if __name__ in {"__main__", "__mp_main__"}:
    main()

