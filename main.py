from datetime import datetime
import time
from data_loader import load_pair_data

def main():
    start_date = "2020-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"Starting correlation analysis from {start_date} to {end_date}")
    tickers = ["GOOG", "META", "NFLX", "MSFT", "AMZN", "TSLA"]
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}")
        print(f"{'='*50}")

        try:
            df = load_pair_data(ticker, start_date, end_date, cache=True)
            if df.empty:
                print(f"No data for {ticker}")
                continue
        
        except Exception as e:
            print(f"Error processing {ticker}")
            continue

if __name__ == "__main__":
    main()
