"""
Feature engineering and model training for AI Disclosure → Trading Lag Predictor.

Builds ~15 interpretable features from event, stock, EDGAR, and GenAI data.
Trains an XGBRegressor to predict peak_car_day (trading days to peak abnormal returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def build_features(events_df: pd.DataFrame,
                   event_study_results: pd.DataFrame,
                   stock_prices: pd.DataFrame,
                   edgar_df: pd.DataFrame,
                   ticker_dim: pd.DataFrame,
                   genai_dim: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from all data sources.

    Returns DataFrame with ~15 features + target (peak_car_day) per event.
    """
    # Start with event study results (has peak_car_day, pre_event metrics)
    df = event_study_results[["event_idx", "ticker", "announcement_date",
                              "peak_car_day", "peak_car_value",
                              "pre_event_vol", "pre_event_avg_volume"]].copy()

    # Drop events where we couldn't compute peak
    df = df.dropna(subset=["peak_car_day"]).copy()
    df["peak_car_day"] = df["peak_car_day"].astype(int)

    # Merge event details
    events_cols = ["ticker", "announcement_date", "use_case", "agent_type",
                   "ai_vendor", "ai_model", "industry", "sp_index"]
    events_subset = events_df[
        [c for c in events_cols if c in events_df.columns]
    ].copy()
    events_subset = events_subset.reset_index().rename(columns={"index": "event_idx"})

    df = df.merge(events_subset, on=["event_idx"], suffixes=("", "_evt"))
    # Use the ticker from event_study_results (already there)
    if "ticker_evt" in df.columns:
        df.drop(columns=["ticker_evt"], inplace=True)
    if "announcement_date_evt" in df.columns:
        df.drop(columns=["announcement_date_evt"], inplace=True)

    # --- Feature 1-2: sector, industry from ticker_dim ---
    if "sector" not in df.columns:
        df = df.merge(ticker_dim[["ticker", "sector", "industry"]],
                      on="ticker", how="left", suffixes=("", "_dim"))
        if "industry_dim" in df.columns:
            df["industry"] = df["industry"].fillna(df["industry_dim"])
            df.drop(columns=["industry_dim"], inplace=True)

    # --- Feature 3-4: use_case, agent_type (already merged) ---

    # --- Feature 5-6: has_ai_vendor, has_ai_model ---
    df["has_ai_vendor"] = df["ai_vendor"].notna().astype(int) if "ai_vendor" in df.columns else 0
    df["has_ai_model"] = df["ai_model"].notna().astype(int) if "ai_model" in df.columns else 0

    # --- Feature 7: prior_event_count ---
    df = df.sort_values("announcement_date").reset_index(drop=True)
    df["prior_event_count"] = df.groupby("ticker").cumcount()

    # --- Feature 8-10: pre_event_volatility, pre_event_avg_volume, pre_event_return_30d ---
    # pre_event_vol and pre_event_avg_volume already from event study
    df.rename(columns={"pre_event_vol": "pre_event_volatility"}, inplace=True)

    # Compute 30-day pre-event return
    pre_returns = []
    stock_prices_sorted = stock_prices.sort_values(["ticker", "Date"])
    for _, row in df.iterrows():
        ticker = row["ticker"]
        ann = pd.Timestamp(row["announcement_date"])
        tk_data = stock_prices_sorted[stock_prices_sorted["ticker"] == ticker]
        pre = tk_data[tk_data["Date"] < ann].tail(30)
        if len(pre) >= 2:
            ret_30d = (pre["Adj_Close"].iloc[-1] / pre["Adj_Close"].iloc[0]) - 1
        else:
            ret_30d = np.nan
        pre_returns.append(ret_30d)
    df["pre_event_return_30d"] = pre_returns

    # --- Feature 11-12: capex_growth, rd_growth ---
    df["capex_growth"] = _compute_fundamental_growth(
        df, edgar_df, concept="CapitalExpenditures"
    )
    df["rd_growth"] = _compute_fundamental_growth(
        df, edgar_df, concept="ResearchAndDevelopmentExpense"
    )

    # --- Feature 13: days_since_genai_release ---
    genai_dates = genai_dim["release_date"].sort_values().values
    days_since = []
    for _, row in df.iterrows():
        ann = pd.Timestamp(row["announcement_date"])
        prior = genai_dates[genai_dates <= ann.to_datetime64()]
        if len(prior) > 0:
            last = pd.Timestamp(prior[-1])
            days_since.append((ann - last).days)
        else:
            days_since.append(np.nan)
    df["days_since_genai_release"] = days_since

    # --- Feature 14-15: event_year, event_quarter ---
    df["event_year"] = pd.to_datetime(df["announcement_date"]).dt.year
    df["event_quarter"] = pd.to_datetime(df["announcement_date"]).dt.quarter

    return df


def _compute_fundamental_growth(df: pd.DataFrame,
                                edgar_df: pd.DataFrame,
                                concept: str) -> list[float]:
    """Compute QoQ growth for a given EDGAR concept for each event."""
    concept_data = edgar_df[edgar_df["concept"] == concept].copy()
    concept_data = concept_data.sort_values(["ticker", "end_date"])

    growth_values = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        ann = pd.Timestamp(row["announcement_date"])
        tk_data = concept_data[
            (concept_data["ticker"] == ticker) &
            (concept_data["end_date"] <= ann)
        ].tail(2)
        if len(tk_data) == 2:
            prev, curr = tk_data["value"].values
            if prev != 0:
                growth_values.append((curr - prev) / abs(prev))
            else:
                growth_values.append(np.nan)
        else:
            growth_values.append(np.nan)
    return growth_values


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["sector", "industry", "use_case", "agent_type"]
NUMERIC_FEATURES = [
    "has_ai_vendor", "has_ai_model", "prior_event_count",
    "pre_event_volatility", "pre_event_avg_volume", "pre_event_return_30d",
    "capex_growth", "rd_growth", "days_since_genai_release",
    "event_year", "event_quarter",
]
TARGET = "peak_car_day"


def prepare_xy(feature_matrix: pd.DataFrame) -> tuple:
    """
    Encode categoricals and split into X, y.

    Returns (X, y, label_encoders) where label_encoders is a dict
    of fitted LabelEncoders for each categorical column.
    """
    df = feature_matrix.copy()
    label_encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna("Unknown")
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    feature_cols = [c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    return X, y, label_encoders


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                **kwargs) -> XGBRegressor:
    """Train an XGBRegressor."""
    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "enable_categorical": False,
    }
    params.update(kwargs)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    # Clip predictions to valid range
    y_pred = np.clip(y_pred, 1, 60)

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "y_pred": y_pred,
        "y_test": y_test.values,
    }


def mean_baseline_mae(y_test: pd.Series) -> float:
    """MAE of a baseline that always predicts the mean."""
    return mean_absolute_error(y_test, np.full(len(y_test), y_test.mean()))


def predict_for_company(model, ticker: str, event_info: dict,
                        stock_prices: pd.DataFrame,
                        edgar_df: pd.DataFrame,
                        ticker_dim: pd.DataFrame,
                        genai_dim: pd.DataFrame,
                        label_encoders: dict,
                        feature_cols: list[str]) -> dict:
    """
    Predict peak CAR lag for a new company event.

    Parameters
    ----------
    model : Fitted XGBRegressor.
    ticker : Stock ticker.
    event_info : Dict with keys like use_case, agent_type, ai_vendor, ai_model,
                 announcement_date.
    stock_prices, edgar_df, ticker_dim, genai_dim : Data sources.
    label_encoders : From prepare_xy().
    feature_cols : List of feature column names in model order.

    Returns
    -------
    Dict with predicted_lag, features used.
    """
    ann_date = pd.Timestamp(event_info.get("announcement_date", pd.Timestamp.now()))

    # Build a single-row feature dict
    features = {}

    # Sector / industry
    tk_info = ticker_dim[ticker_dim["ticker"] == ticker]
    features["sector"] = tk_info["sector"].values[0] if len(tk_info) > 0 else "Unknown"
    features["industry"] = tk_info["industry"].values[0] if len(tk_info) > 0 else "Unknown"

    # Event details
    features["use_case"] = event_info.get("use_case", "Unknown")
    features["agent_type"] = event_info.get("agent_type", "Unknown")
    features["has_ai_vendor"] = 1 if event_info.get("ai_vendor") else 0
    features["has_ai_model"] = 1 if event_info.get("ai_model") else 0

    # Prior event count (0 for new prediction)
    features["prior_event_count"] = event_info.get("prior_event_count", 0)

    # Pre-event stock metrics
    tk_prices = stock_prices[stock_prices["ticker"] == ticker].sort_values("Date")
    pre = tk_prices[tk_prices["Date"] < ann_date].tail(30)
    if len(pre) >= 2:
        returns = np.log(pre["Adj_Close"] / pre["Adj_Close"].shift(1)).dropna()
        features["pre_event_volatility"] = returns.std()
        features["pre_event_avg_volume"] = pre["Volume"].mean()
        features["pre_event_return_30d"] = (pre["Adj_Close"].iloc[-1] / pre["Adj_Close"].iloc[0]) - 1
    else:
        features["pre_event_volatility"] = np.nan
        features["pre_event_avg_volume"] = np.nan
        features["pre_event_return_30d"] = np.nan

    # EDGAR fundamentals
    for concept, feat_name in [("CapitalExpenditures", "capex_growth"),
                                ("ResearchAndDevelopmentExpense", "rd_growth")]:
        concept_data = edgar_df[
            (edgar_df["ticker"] == ticker) &
            (edgar_df["concept"] == concept) &
            (edgar_df["end_date"] <= ann_date)
        ].sort_values("end_date").tail(2)
        if len(concept_data) == 2:
            prev, curr = concept_data["value"].values
            features[feat_name] = (curr - prev) / abs(prev) if prev != 0 else np.nan
        else:
            features[feat_name] = np.nan

    # Days since last GenAI release
    genai_dates = genai_dim["release_date"].sort_values().values
    prior = genai_dates[genai_dates <= ann_date.to_datetime64()]
    features["days_since_genai_release"] = (ann_date - pd.Timestamp(prior[-1])).days if len(prior) > 0 else np.nan

    # Time features
    features["event_year"] = ann_date.year
    features["event_quarter"] = (ann_date.month - 1) // 3 + 1

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        if col in label_encoders and col in features:
            le = label_encoders[col]
            val = features[col]
            if val in le.classes_:
                features[col] = le.transform([val])[0]
            else:
                features[col] = -1  # unknown category

    # Build feature vector in correct order
    feature_vector = pd.DataFrame([features])[feature_cols]
    prediction = model.predict(feature_vector)[0]
    prediction = int(np.clip(round(prediction), 1, 60))

    return {
        "predicted_lag": prediction,
        "ticker": ticker,
        "announcement_date": ann_date,
        "features": features,
        "feature_vector": feature_vector,
    }
