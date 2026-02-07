import os
import pandas as pd
import yfinance as yf

def get_data_path(ticker, start_date, end_date, folder="data/raw"):
    filename = f"{ticker}_{start_date}_to_{end_date}_raw.csv"
    return os.path.join(folder, filename)


def get_stock_data(ticker, start_date, end_date, cache=True):
    path = get_data_path(ticker, start_date, end_date)
    if cache and os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])

    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df = df[["Date", "Close"]]
    df["Ticker"] = ticker
    if cache:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv(path, index=False)
    
    return df

def load_pair_data(ticker, start_date, end_date, cache=True):
    """
    Load and align historical price data for a pair of stocks.

    Args:
        ticker_a (str): First stock ticker (e.g., 'AAPL').
        ticker_b (str): Second stock ticker (e.g., 'MSFT').
        start_date (str): Start date for both stocks.
        end_date (str): End date for both stocks.
        cache (bool): Whether to cache individual stock data locally.

    Returns:
        pd.DataFrame: Merged DataFrame with aligned 'Date', 'Close_<ticker>' columns.
    """
    print(f"Loading data for {ticker}")
    df = get_stock_data(ticker, start_date, end_date, cache)
    
    return df


