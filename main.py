from datetime import datetime
import time
from data_loader import load_pair_data

def main():
    start_date = "2020-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"Starting correlation analysis from {start_date} to {end_date}")
    tickers = ["NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","META","AVGO","TSLA","BRK.B","WMT","LLY","JPM","V","XOM","JNJ","MA","COST","MU","ORCL","BAC","ABBV","HD","PG","CVX","NFLX","KO","CAT","AMD","GE","CSCO","PLTR","MRK","WFC","LRCX","MS","PM","IBM","GS","RTX","AMAT","INTC","UNH","AXP","PEP","MCD","TMUS","C","GEV","LIN","AMGN","TMO","TXN","VZ","ABT","DIS","T","BA","GILD","KLAC","SCHW","NEE","CRM","ISRG","ANET","TJX"]
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
