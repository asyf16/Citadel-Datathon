"""
Event study module for AI Disclosure → Trading Lag Predictor.

Computes abnormal returns (market-adjusted model), cumulative abnormal
returns (CAR), and detects the peak CAR day for each AI disclosure event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices_df: pd.DataFrame,
                    price_col: str = "Adj_Close",
                    group_col: str | None = "ticker") -> pd.DataFrame:
    """
    Compute daily log returns per ticker (or for a single series).

    Parameters
    ----------
    prices_df : DataFrame with Date, price_col, and optionally group_col.
    price_col : Column containing prices.
    group_col : Column to group by (None for single-series like SPY).

    Returns
    -------
    DataFrame with added 'log_return' column.
    """
    df = prices_df.sort_values(
        ["Date"] if group_col is None else [group_col, "Date"]
    ).copy()

    if group_col:
        df["log_return"] = df.groupby(group_col)[price_col].transform(
            lambda x: np.log(x / x.shift(1))
        )
    else:
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    return df


def compute_abnormal_returns(stock_ret: pd.Series,
                             spy_ret: pd.Series) -> pd.Series:
    """
    Abnormal return = stock daily return - SPY daily return.
    Simple market-adjusted model (no OLS estimation needed).
    """
    return stock_ret - spy_ret


def find_peak_car_day(ar_series: pd.Series, window: int = 60,
                      rolling_w: int = 5) -> int:
    """
    Find the day in [1, window] with peak abnormal return activity.

    Uses a rolling-window average of daily ARs to detect the period of
    maximum abnormal return concentration, which is more robust to drift
    than raw cumulative AR.

    Parameters
    ----------
    ar_series : Series of daily abnormal returns indexed by relative day.
    window : Maximum days after event to search.
    rolling_w : Rolling window size for smoothing.

    Returns
    -------
    int : Day number (1-based) with peak rolling |AR|.
    """
    post_event = ar_series.loc[
        (ar_series.index >= 1) & (ar_series.index <= window)
    ]
    if post_event.empty:
        return window  # fallback

    # Use rolling mean of absolute AR to find the peak activity period
    rolling_abs = post_event.abs().rolling(rolling_w, min_periods=1, center=True).mean()
    return int(rolling_abs.idxmax())


def _get_event_window(stock_returns_df: pd.DataFrame,
                      spy_returns_df: pd.DataFrame,
                      ticker: str,
                      event_date: pd.Timestamp,
                      pre_days: int = 30,
                      post_days: int = 60) -> dict | None:
    """
    Extract returns around a single event and compute CAR + peak.

    Returns dict with peak_car_day, peak_car_value, pre_event_vol,
    pre_event_avg_volume, or None if insufficient data.
    """
    # Get stock data for this ticker
    ticker_data = stock_returns_df[
        stock_returns_df["ticker"] == ticker
    ].set_index("Date").sort_index()

    spy_data = spy_returns_df.set_index("Date").sort_index()

    if ticker_data.empty:
        return None

    # Find the closest trading date to the event date
    trading_dates = ticker_data.index
    idx = trading_dates.searchsorted(event_date)
    if idx >= len(trading_dates):
        idx = len(trading_dates) - 1

    # Snap to nearest trading date
    if idx > 0 and abs(trading_dates[idx] - event_date) > abs(trading_dates[idx - 1] - event_date):
        idx = idx - 1
    event_trading_date = trading_dates[idx]

    # Define window bounds
    start_idx = max(0, idx - pre_days)
    end_idx = min(len(trading_dates) - 1, idx + post_days)

    if end_idx - idx < 10:  # need at least 10 post-event days
        return None

    window_dates = trading_dates[start_idx:end_idx + 1]
    window_data = ticker_data.loc[window_dates].copy()

    # Align SPY returns
    spy_aligned = spy_data.reindex(window_dates)
    spy_returns = spy_aligned["log_return"].fillna(0)

    # Compute abnormal returns
    stock_returns = window_data["log_return"].fillna(0)
    ar = compute_abnormal_returns(stock_returns, spy_returns)

    # Create relative day index (0 = event day)
    relative_days = pd.Series(
        range(-(idx - start_idx), end_idx - idx + 1),
        index=window_dates
    )
    ar.index = relative_days.values

    # Cumulative abnormal returns from day 0
    post_event_ar = ar.loc[ar.index >= 0]
    car = post_event_ar.cumsum()

    # Peak CAR day — use daily AR series for robust peak detection
    peak_day = find_peak_car_day(ar, window=post_days)
    peak_value = car.loc[peak_day] if peak_day in car.index else 0.0

    # Pre-event metrics
    pre_event = window_data.iloc[:idx - start_idx]
    pre_event_vol = pre_event["log_return"].std() if len(pre_event) > 2 else np.nan
    pre_event_avg_volume = pre_event["Volume"].mean() if "Volume" in pre_event.columns else np.nan

    # Full AR series for plotting
    ar_series = ar.to_dict()
    car_full = ar.sort_index().cumsum()
    car_series = car_full.to_dict()

    return {
        "peak_car_day": peak_day,
        "peak_car_value": peak_value,
        "pre_event_vol": pre_event_vol,
        "pre_event_avg_volume": pre_event_avg_volume,
        "ar_series": ar_series,
        "car_series": car_series,
    }


def run_event_study(events_df: pd.DataFrame,
                    stock_prices: pd.DataFrame,
                    spy_prices: pd.DataFrame,
                    pre_days: int = 30,
                    post_days: int = 60) -> pd.DataFrame:
    """
    Run event study over all AI disclosure events.

    Parameters
    ----------
    events_df : DataFrame with ticker, announcement_date columns.
    stock_prices : Daily stock prices (Date, Adj_Close, Volume, ticker).
    spy_prices : SPY daily prices (Date, Adj_Close).

    Returns
    -------
    DataFrame with one row per event, including peak_car_day, peak_car_value,
    pre_event_vol, pre_event_avg_volume columns.
    """
    # Compute returns once
    stock_ret = compute_returns(stock_prices, price_col="Adj_Close", group_col="ticker")
    spy_ret = compute_returns(spy_prices, price_col="Adj_Close", group_col=None)

    results = []
    ar_curves = []
    car_curves = []

    for i, row in events_df.iterrows():
        ticker = row["ticker"]
        ann_date = pd.Timestamp(row["announcement_date"])

        window_result = _get_event_window(
            stock_ret, spy_ret, ticker, ann_date,
            pre_days=pre_days, post_days=post_days
        )

        if window_result is None:
            results.append({
                "event_idx": i,
                "ticker": ticker,
                "announcement_date": ann_date,
                "peak_car_day": np.nan,
                "peak_car_value": np.nan,
                "pre_event_vol": np.nan,
                "pre_event_avg_volume": np.nan,
            })
        else:
            results.append({
                "event_idx": i,
                "ticker": ticker,
                "announcement_date": ann_date,
                "peak_car_day": window_result["peak_car_day"],
                "peak_car_value": window_result["peak_car_value"],
                "pre_event_vol": window_result["pre_event_vol"],
                "pre_event_avg_volume": window_result["pre_event_avg_volume"],
            })
            ar_curves.append(window_result["ar_series"])
            car_curves.append(window_result["car_series"])

    result_df = pd.DataFrame(results)

    # Store curves as attributes for plotting
    result_df.attrs["ar_curves"] = ar_curves
    result_df.attrs["car_curves"] = car_curves

    return result_df


def get_average_car_curve(event_study_df: pd.DataFrame,
                          min_day: int = -5,
                          max_day: int = 60) -> pd.DataFrame:
    """
    Compute average CAR curve across all events for plotting.

    Returns DataFrame with columns: day, mean_car, std_car, count.
    """
    car_curves = event_study_df.attrs.get("car_curves", [])
    if not car_curves:
        return pd.DataFrame(columns=["day", "mean_car", "std_car", "count"])

    # Collect all CAR values by relative day
    day_values = {}
    for curve in car_curves:
        for day, val in curve.items():
            if min_day <= day <= max_day:
                day_values.setdefault(day, []).append(val)

    rows = []
    for day in sorted(day_values.keys()):
        vals = day_values[day]
        rows.append({
            "day": day,
            "mean_car": np.mean(vals),
            "std_car": np.std(vals),
            "count": len(vals),
        })

    return pd.DataFrame(rows)
