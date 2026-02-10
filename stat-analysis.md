# Plan: Statistical Analysis — Market Reactions to AI Adoption Disclosures

## Context

The existing `model.md` focuses on a predictive lag model (days to peak CAR). This is a **separate, complementary analysis**: an empirical study testing whether AI adoption announcements affect trading intensity and volatility, whether AI-related CapEx amplifies those effects, and whether the effects have attenuated over time (2015→2025) as AI became mainstream.

**Core hypothesis:** Early AI announcements (e.g., 2016) triggered outsized market reactions (volume spikes, volatility jumps), but as AI became commonplace, the same type of disclosure now generates a muted response.

---

## Deliverables

1. **`src/statistical_analysis.py`** — helper module with all reusable functions
2. **`statistical_analysis.ipynb`** — standalone notebook (separate from `main.ipynb`)

---

## Step 1: Create `src/statistical_analysis.py`

### 1A. Data loading & utilities

| Function | Description |
|----------|-------------|
| `load_analysis_data(data_dir)` | → dict of DataFrames (events, prices, spy_prices, edgar, ticker_dim, genai_dim). Falls back to `mock_data.py` if CSVs absent. Concatenates sp500 + sp400 + etfs price files; extracts SPY as benchmark. |
| `assign_period_bin(date)` | → one of `'2015-2017'`, `'2018-2020'`, `'2021-2023'`, `'2024-2025'` |
| `get_event_window(prices_df, ticker, event_date, pre_days=30, post_days=60)` | → DataFrame with `rel_day` column. Handles weekends/holidays via searchsorted. |

### 1B. Trading intensity (abnormal volume)

| Function | Description |
|----------|-------------|
| `compute_event_abnormal_volume_stats(prices_df, events_df, windows=[(-1,1),(0,5),(0,10),(0,20)], pre_window=(-30,-5))` | → DataFrame with columns `abvol_-1_1`, `abvol_0_5`, etc. |

- **Abnormal Volume** = Volume_t / mean(Volume over pre_window)

### 1C. Volatility

| Function | Description |
|----------|-------------|
| `compute_event_volatility_stats(prices_df, spy_prices, events_df, pre_window=(-30,-5), post_window=(1,30))` | → DataFrame with `pre_vol`, `post_vol`, `vol_change`, `vol_ratio`, `max_drawdown`, `drawdown_speed` |
| `compute_max_drawdown(cumulative_returns)` | → (max_drawdown, days_to_trough) |

- **Realized vol** = std(daily returns) in window
- **Drawdown speed** = max_drawdown / days_to_trough
- Reuses `compute_returns()` from `src/event_study.py`

### 1D. CapEx matching

| Function | Description |
|----------|-------------|
| `match_capex_to_events(events_df, edgar_df)` | → merged DataFrame. For each (ticker, event_date), finds most recent EDGAR quarter where `end_date < announcement_date` (no look-ahead bias). Pivots concepts to columns, computes: `capex_intensity` = PaymentsToAcquirePropertyPlantAndEquipment / PropertyPlantAndEquipmentNet; `rd_intensity` = ResearchAndDevelopmentExpense (level). |
| `classify_capex_high_low(df, column='capex_intensity')` | → adds `high_capex` binary column (median split) |

### 1E. Statistical tests

| Function | Test | Null |
|----------|------|------|
| `test_abnormal_volume_significance(vals)` | One-sample t-test + Wilcoxon (one-sided) | AbVol = 1 |
| `test_paired_volatility_change(pre, post)` | Paired t-test + Wilcoxon | pre_vol = post_vol |
| `test_period_differences(df, value_col, group_col)` | ANOVA + Kruskal-Wallis + pairwise Mann-Whitney (Bonferroni) | All period means equal |
| `test_capex_group_difference(high, low)` | Two-sample t-test + Mann-Whitney U | High CapEx = Low CapEx |

### 1F. Regression

| Function | Description |
|----------|-------------|
| `run_ols_regression(df, dependent_var, independent_vars, add_period_dummies=True)` | → statsmodels OLS result. Period dummies via `pd.get_dummies(drop_first=True)`. Adds constant via `sm.add_constant()`. |

### 1G. Visualization helpers

- `plot_abnormal_volume_by_period(stats_df, window_col)` — box + strip plot
- `plot_volatility_change_by_period(vol_df)` — box + strip plot
- `plot_capex_interaction(df, reaction_col)` — grouped bar (high/low CapEx by period)
- `plot_time_trend_line(df, y_col)` — line with 95% CI error bars
- `format_test_results_table(results)` — dict → formatted DataFrame with significance stars

---

## Step 2: Create `statistical_analysis.ipynb`

### Notebook structure (27 cells)

| Section | Cells | Content |
|---------|-------|---------|
| **Title + Setup** | 0–1 | Imports, path setup, plot styling. Import from `src/statistical_analysis` and `src/event_study` |
| **Data Loading** | 2 | Load all data, add period_bin, merge sector info, print summary |
| **1. EDA** | 3–4 | Events by year (bar), by sector (barh), by agent_type (bar) — 3-panel figure |
| **2. Trading Intensity** | 5–8 | Compute abnormal volume → t-test/Wilcoxon for each window → 4-panel histogram of AbVol distributions |
| **3. Volatility** | 9–12 | Compute pre/post vol → paired t-test → 3-panel: scatter (pre vs post), hist (vol change), hist (drawdown) |
| **4. Time Trend** | 13–17 | AbVol by period (box + line with CI) → ANOVA/Kruskal + pairwise tests → Vol change by period (box + line) → period tests |
| **5. CapEx Interaction** | 18–22 | Match EDGAR → median split → high vs low group comparison → grouped bar + scatter → OLS regressions (AbVol ~ CapEx + period dummies; VolChange ~ CapEx + period dummies) |
| **6. Cross-Sectional** | 23–24 | 2×2 grid: mean AbVol by agent_type, ai_vendor, sector, use_case (horizontal bars with CI) |
| **7. Summary** | 25–27 | Summary table of all test results with p-values → narrative conclusions → limitations |

### Key regression models

- **Model 1:** AbVol[0,+5] ~ capex_intensity + rd_intensity + period_dummies
- **Model 2:** VolChange ~ capex_intensity + rd_intensity + period_dummies

---

## Step 3: Dependencies & sequencing

1. **`src/event_study.py`** and **`src/mock_data.py`** must exist first (for `compute_returns` and mock data fallback). If they don’t exist yet, include minimal standalone versions of needed functions in `statistical_analysis.py`.
2. Build **`src/statistical_analysis.py`** (data loading → volume → volatility → CapEx → tests → regression → viz helpers).
3. Build **`statistical_analysis.ipynb`** consuming the module.

---

## Design considerations

- **Small samples per bin:** ~200 events ÷ 4 bins → some bins may have <20 events. Use non-parametric tests as primary; report group Ns.
- **Overlapping events:** Multiple AI announcements per ticker may overlap windows. Flag these; optionally exclude in robustness check.
- **Weekend/holiday dates:** `get_event_window` uses searchsorted to snap to next trading day.
- **Missing EDGAR data:** Report match rate; CapEx analysis uses only matched subset.
- **Style:** `sns.set_theme(style="whitegrid")`. Steelblue for volume, coral for volatility, navy for CapEx. All figures saved to `outputs/figures/` at 150 DPI.

---

## Verification

1. Notebook runs top-to-bottom with mock data (Kernel → Restart & Run All).
2. All statistical tests produce p-values (no NaN/errors).
3. All 7+ figures render with labels and legends.
4. CapEx matching has no look-ahead bias (assert `end_date < announcement_date` for all matches).
5. Period bin assignment covers all dates in [2015, 2025].
6. Regression summaries display via `model.summary()`.
