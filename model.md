# Plan: AI Disclosure → Trading Lag Predictor (Simplified)

## Context

These were the instructions provided to Claude.

**Question:** When a company discloses AI adoption, how many days until the stock sees peak abnormal returns?

**Goal:** Predictive model — input (ticker, event details) → output (predicted lag in days).

**Example:** Meta announces new AI release → model predicts peak returns in ~5 trading days.

**Constraint:** Completable in ~7 hours. Mock data so it runs now; swap to real CSVs later.

---

## Simplified Structure

```
cit-datathon/predictive-model/
├── main.ipynb                # Single notebook — all analysis, model, demo
├── src/
│   ├── __init__.py
│   ├── mock_data.py          # Generate mock DataFrames matching CSV schemas
│   ├── event_study.py        # Abnormal returns + peak detection (simplified)
│   └── features.py           # Feature engineering + model training
├── tests/
│   └── test_pipeline.py      # Pytest test cases proving the pipeline works
├── data/                     # Drop real CSVs here when available
└── outputs/figures/          # Saved plots
```

4 source files + 1 notebook + 1 test file. That's it.

---

## Step 1: Mock Data Generator (`src/mock_data.py`)

Create realistic mock data matching these schemas:

| Mock Dataset   | Rows  | Key Columns |
|----------------|-------|-------------|
| events         | ~200  | ticker, announcement_date, use_case, agent_type, ai_vendor, ai_model, industry, sp_index |
| stock_prices   | ~50K  | Date, Open, High, Low, Close, Adj_Close, Volume, ticker (~50 tickers × ~2500 days) |
| spy_prices     | ~2500 | Date, Adj_Close (SPY benchmark) |
| edgar_capex    | ~2K   | ticker, concept, value, end_date, fiscal_year, fiscal_period |
| ticker_dim     | ~50   | ticker, sector, industry |
| genai_releases | ~30   | model_name, release_date, company |

- Stock prices use **geometric Brownian motion**.
- Events inject small return bumps at random lags (5–40 days) so the model has a learnable signal. This proves the pipeline works end-to-end.

**Real data toggle:** `data_loader()` checks if CSVs exist in `data/` → uses real data; otherwise → mock.

---

## Step 2: Event Study — Simplified (`src/event_study.py`)

For each AI disclosure event:

1. Get stock returns around the event (−30 to +60 trading days).
2. Get SPY returns for same window.
3. **Abnormal return** = stock daily return − SPY daily return (simple market-adjusted model, no OLS estimation needed).
4. **Cumulative abnormal return (CAR)** = running sum of daily ARs from event day.
5. **Peak CAR day** = which day in [+1, +60] has max |CAR| → this is the **target variable**.
6. **Abnormal volume** = daily volume / avg volume from pre-event 30 days.

### Functions

| Function | Description |
|----------|-------------|
| `compute_returns(prices_df)` | → daily log returns per ticker |
| `compute_abnormal_returns(stock_ret, spy_ret)` | → AR = stock − market |
| `find_peak_car_day(car_series, window=60)` | → int (days to peak) |
| `run_event_study(events_df, stock_prices, spy_prices)` | → DataFrame with peak_car_day + auxiliary metrics per event |

**Output:** ~200 rows (one per event) with columns: `ticker`, `announcement_date`, `peak_car_day`, `peak_car_value`, `pre_event_vol`, `pre_event_avg_volume`.

---

## Step 3: Features + Model (`src/features.py`)

~15 features (manageable, interpretable):

| # | Feature | Source | Type |
|---|---------|--------|------|
| 1 | sector | ticker_dim | categorical (encoded) |
| 2 | industry | ticker_dim | categorical (encoded) |
| 3 | use_case | events | categorical (encoded) |
| 4 | agent_type | events | categorical (encoded) |
| 5 | has_ai_vendor | events | binary (1 if vendor named) |
| 6 | has_ai_model | events | binary (1 if model named) |
| 7 | prior_event_count | derived | int (how many prior AI events for this ticker) |
| 8 | pre_event_volatility | stock prices | float (30-day return std before event) |
| 9 | pre_event_avg_volume | stock prices | float (30-day avg volume before event) |
| 10 | pre_event_return_30d | stock prices | float (cumulative return 30d before) |
| 11 | capex_growth | EDGAR | float (QoQ CapEx growth, latest before event) |
| 12 | rd_growth | EDGAR | float (QoQ R&D growth) |
| 13 | days_since_genai_release | genai_dim | int (days since last major model release) |
| 14 | event_year | derived | int |
| 15 | event_quarter | derived | int |

### Functions

| Function | Description |
|----------|-------------|
| `build_features(events_df, event_study_results, stock_prices, edgar_df, ticker_dim, genai_dim)` | → feature matrix DataFrame |
| `prepare_xy(feature_matrix)` | → X (encoded), y (peak_car_day) |
| `train_model(X_train, y_train)` | → fitted XGBRegressor |
| `evaluate_model(model, X_test, y_test)` | → metrics dict (MAE, RMSE, R²) |
| `predict_for_company(model, ticker, event_info, ...)` | → predicted lag in days |

- **Model:** XGBRegressor from `xgboost` — `model.fit(X_train, y_train)` / `model.predict(X_test)`.
- **Split:** Time-based (80% train / 20% test by `announcement_date`).
- **Encoding:** LabelEncoder for categoricals; XGBoost handles NaN natively.

---

## Step 4: Single Notebook (`main.ipynb`)

### Section 1: Setup & Data Loading (~30 min)

- Import libraries (pandas, numpy, xgboost, sklearn, shap, matplotlib, seaborn).
- Load data (mock or real).
- Quick EDA: event count by year, top sectors, top AI vendors.

### Section 2: Event Study (~1 hr)

- Compute returns for all stocks + SPY.
- Run event study over all events.
- **Plot 1:** Average CAR curve (−5 to +60 days) with confidence band — shows how the market digests AI news over time.
- **Plot 2:** Distribution of `peak_car_day` — histogram showing when peaks happen.
- Statistical test: t-test on mean AR at day 0 (is event-day return significant?).

### Section 3: Feature Engineering (~1 hr)

- Build feature matrix.
- Feature correlation heatmap.
- Missing value summary.

### Section 4: Model Training & Evaluation (~1.5 hr)

- Time-based train/test split.
- Train baselines (mean predictor, Ridge regression).
- Train XGBoost with cross-validation.
- **Plot 3:** Model comparison bar chart (MAE).
- **Plot 4:** Actual vs Predicted scatter.
- **Plot 5:** SHAP feature importance (bar plot + beeswarm).
- Print metrics table.

### Section 5: Demo — Predict for Meta (~30 min)

- Call `predict_for_company('META', {...})`.
- Show: *"Predicted: peak returns in X days after disclosure"*.
- **Plot 6:** SHAP force plot for the Meta prediction.
- **Plot 7:** Meta's historical AI events with CAR curves.

### Section 6: Conclusions

- Key findings summary.
- Limitations & future work.

---

## Step 5: Test Cases (`tests/test_pipeline.py`)

| # | Test | Description |
|---|------|-------------|
| 1 | `test_mock_data_shapes` | Mock data generates correct shapes and columns |
| 2 | `test_compute_returns` | Returns computation produces valid values (no NaN for interior dates) |
| 3 | `test_peak_car_day_in_range` | Event study finds peak within [1, 60] range |
| 4 | `test_no_target_leakage` | Feature matrix has no target leakage (no post-event features) |
| 5 | `test_model_train_predict` | Model trains and predicts without error |
| 6 | `test_model_beats_baseline` | Model beats mean baseline on mock data |
| 7 | `test_predict_for_company` | `predict_for_company` returns valid integer in [1, 60] |
| 8 | `test_end_to_end` | End-to-end pipeline runs from mock data to prediction |

Run with:

```bash
pytest tests/test_pipeline.py -v
```

---

## Dependencies

```
pandas numpy scikit-learn xgboost shap matplotlib seaborn scipy pytest
```

All pip-installable. No PyTorch, no TensorFlow.

---

## Real Data Swap

When CSVs are downloaded, drop them in `data/`:

- `enterprise_ai_adoption_internet_events.csv`
- `sp500_prices_all_since_2015.csv`
- `sp400_prices_all_since_2015.csv`
- `etfs_prices_all_since_2015.csv`
- `sp_edgar_fundamentals.csv`
- `ticker_dimension.csv` (or `ticker_dimens.csv`)
- `genai_dimension.csv`

The `data_loader()` auto-detects real files and switches from mock. Column names match the schemas from the datathon packet exactly.

---

## Time Budget (7 hours)

| Step | Time | Deliverable |
|------|------|-------------|
| Mock data + data loader | 1 hr | src/mock_data.py, data loading in notebook |
| Event study module + notebook section | 1.5 hr | src/event_study.py, CAR plots |
| Feature engineering | 1 hr | src/features.py, feature matrix |
| Model training + evaluation | 1.5 hr | XGBoost trained, SHAP plots, metrics |
| Demo + visualizations | 1 hr | Meta prediction, all 7 plots |
| Tests + polish | 1 hr | tests/test_pipeline.py, clean notebook |

---

## Verification

1. **`pytest tests/test_pipeline.py -v`** — all 8 tests pass.
2. **Notebook runs top-to-bottom** with mock data (Kernel → Restart & Run All).
3. **Model MAE < mean baseline MAE** on mock data.
4. **Meta demo** outputs a prediction between 1–60 days with SHAP explanation.
5. **All 7 plots** render with labels and legends.
