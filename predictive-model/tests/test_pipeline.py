"""
Test suite for AI Disclosure → Trading Lag Predictor pipeline.

Run with: pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mock_data import (
    data_loader,
    generate_stock_prices,
    generate_spy_prices,
    generate_events,
    generate_edgar_capex,
    generate_ticker_dim,
    generate_genai_releases,
    TICKERS,
)
from src.event_study import (
    compute_returns,
    find_peak_car_day,
    run_event_study,
)
from src.features import (
    build_features,
    prepare_xy,
    train_model,
    evaluate_model,
    mean_baseline_mae,
    predict_for_company,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_data():
    """Load all mock data once for the test module."""
    return data_loader(seed=42)


@pytest.fixture(scope="module")
def event_study_results(mock_data):
    """Run event study once for all tests."""
    return run_event_study(
        mock_data["events"],
        mock_data["stock_prices"],
        mock_data["spy_prices"],
    )


@pytest.fixture(scope="module")
def feature_matrix(mock_data, event_study_results):
    """Build feature matrix once."""
    return build_features(
        mock_data["events"],
        event_study_results,
        mock_data["stock_prices"],
        mock_data["edgar_capex"],
        mock_data["ticker_dim"],
        mock_data["genai_releases"],
    )


@pytest.fixture(scope="module")
def trained_model(feature_matrix):
    """Train model once."""
    X, y, encoders = prepare_xy(feature_matrix)
    # Time-based split
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = train_model(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, encoders, list(X.columns)


# ---------------------------------------------------------------------------
# Test 1: Mock data generates correct shapes and columns
# ---------------------------------------------------------------------------
def test_mock_data_shapes(mock_data):
    events = mock_data["events"]
    stock = mock_data["stock_prices"]
    spy = mock_data["spy_prices"]
    edgar = mock_data["edgar_capex"]
    ticker_dim = mock_data["ticker_dim"]
    genai = mock_data["genai_releases"]

    # Events
    assert len(events) == 200
    assert "ticker" in events.columns
    assert "announcement_date" in events.columns
    assert "use_case" in events.columns

    # Stock prices: ~50 tickers
    assert stock["ticker"].nunique() == len(TICKERS)
    assert len(stock) > 10_000
    for col in ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "ticker"]:
        assert col in stock.columns

    # SPY
    assert "Date" in spy.columns
    assert "Adj_Close" in spy.columns
    assert len(spy) > 2000

    # EDGAR
    assert "ticker" in edgar.columns
    assert "concept" in edgar.columns
    assert len(edgar) > 1000

    # Ticker dim
    assert len(ticker_dim) == len(TICKERS)
    assert "sector" in ticker_dim.columns

    # GenAI releases
    assert len(genai) > 20
    assert "model_name" in genai.columns


# ---------------------------------------------------------------------------
# Test 2: Returns computation produces valid values
# ---------------------------------------------------------------------------
def test_compute_returns(mock_data):
    stock = mock_data["stock_prices"]
    spy = mock_data["spy_prices"]

    stock_ret = compute_returns(stock, price_col="Adj_Close", group_col="ticker")
    spy_ret = compute_returns(spy, price_col="Adj_Close", group_col=None)

    assert "log_return" in stock_ret.columns
    assert "log_return" in spy_ret.columns

    # First row per ticker should be NaN, rest should be valid
    for ticker in stock_ret["ticker"].unique()[:5]:
        tk = stock_ret[stock_ret["ticker"] == ticker].sort_values("Date")
        assert pd.isna(tk["log_return"].iloc[0])
        # Interior values should be finite
        interior = tk["log_return"].iloc[1:]
        assert interior.notna().sum() > 100
        assert np.all(np.isfinite(interior.dropna().values))


# ---------------------------------------------------------------------------
# Test 3: Event study finds peak within [1, 60] range
# ---------------------------------------------------------------------------
def test_peak_car_day_in_range(event_study_results):
    valid = event_study_results.dropna(subset=["peak_car_day"])
    assert len(valid) > 100  # Most events should produce results
    assert (valid["peak_car_day"] >= 1).all()
    assert (valid["peak_car_day"] <= 60).all()


# ---------------------------------------------------------------------------
# Test 4: Feature matrix has no target leakage
# ---------------------------------------------------------------------------
def test_no_target_leakage(feature_matrix):
    # Ensure no post-event features (peak_car_value is allowed as it's
    # derived alongside the target, but no future price data)
    feature_cols = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    actual_features = set(feature_matrix.columns) & feature_cols

    # None of the feature names should suggest post-event data
    post_event_keywords = ["post_event", "future", "forward", "next_day"]
    for col in actual_features:
        for kw in post_event_keywords:
            assert kw not in col.lower(), f"Potential leakage in feature: {col}"

    # Verify pre_event features are computed from pre-event data only
    assert "pre_event_volatility" in feature_matrix.columns
    assert "pre_event_avg_volume" in feature_matrix.columns
    assert "pre_event_return_30d" in feature_matrix.columns


# ---------------------------------------------------------------------------
# Test 5: Model trains and predicts without error
# ---------------------------------------------------------------------------
def test_model_train_predict(trained_model):
    model, X_train, X_test, y_train, y_test, _, _ = trained_model

    # Model should exist and be trained
    assert model is not None
    assert hasattr(model, "predict")

    # Predictions should be numeric
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Test 6: Model beats mean baseline on mock data
# ---------------------------------------------------------------------------
def test_model_beats_baseline(trained_model):
    model, _, X_test, _, y_test, _, _ = trained_model

    metrics = evaluate_model(model, X_test, y_test)
    baseline_mae = mean_baseline_mae(y_test)

    assert metrics["MAE"] < baseline_mae, (
        f"Model MAE ({metrics['MAE']:.2f}) should beat baseline ({baseline_mae:.2f})"
    )


# ---------------------------------------------------------------------------
# Test 7: predict_for_company returns valid integer in [1, 60]
# ---------------------------------------------------------------------------
def test_predict_for_company(trained_model, mock_data):
    model, _, _, _, _, encoders, feature_cols = trained_model

    result = predict_for_company(
        model=model,
        ticker="META",
        event_info={
            "use_case": "Content Creation",
            "agent_type": "Copilot",
            "ai_vendor": "Meta AI",
            "ai_model": "Llama",
            "announcement_date": "2025-01-15",
            "prior_event_count": 3,
        },
        stock_prices=mock_data["stock_prices"],
        edgar_df=mock_data["edgar_capex"],
        ticker_dim=mock_data["ticker_dim"],
        genai_dim=mock_data["genai_releases"],
        label_encoders=encoders,
        feature_cols=feature_cols,
    )

    assert "predicted_lag" in result
    assert 1 <= result["predicted_lag"] <= 60
    assert isinstance(result["predicted_lag"], int)
    assert result["ticker"] == "META"


# ---------------------------------------------------------------------------
# Test 8: End-to-end pipeline runs from mock data to prediction
# ---------------------------------------------------------------------------
def test_end_to_end():
    """Full pipeline: mock data → event study → features → model → prediction."""
    # 1. Load data
    data = data_loader(seed=99)

    # 2. Run event study
    es_results = run_event_study(
        data["events"], data["stock_prices"], data["spy_prices"]
    )
    assert len(es_results) > 0

    # 3. Build features
    fm = build_features(
        data["events"], es_results, data["stock_prices"],
        data["edgar_capex"], data["ticker_dim"], data["genai_releases"]
    )
    assert TARGET in fm.columns
    assert len(fm) > 50

    # 4. Prepare and train
    X, y, encoders = prepare_xy(fm)
    n = len(X)
    split = int(n * 0.8)
    model = train_model(X.iloc[:split], y.iloc[:split])

    # 5. Evaluate
    metrics = evaluate_model(model, X.iloc[split:], y.iloc[split:])
    assert metrics["MAE"] > 0
    assert np.isfinite(metrics["RMSE"])

    # 6. Predict
    result = predict_for_company(
        model=model,
        ticker="AAPL",
        event_info={
            "use_case": "Predictive Analytics",
            "agent_type": "Analytics",
            "ai_vendor": "OpenAI",
            "ai_model": "GPT-4",
            "announcement_date": "2025-03-01",
        },
        stock_prices=data["stock_prices"],
        edgar_df=data["edgar_capex"],
        ticker_dim=data["ticker_dim"],
        genai_dim=data["genai_releases"],
        label_encoders=encoders,
        feature_cols=list(X.columns),
    )
    assert 1 <= result["predicted_lag"] <= 60
