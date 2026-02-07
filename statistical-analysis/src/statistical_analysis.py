"""
Statistical analysis module for AI Disclosure market reaction study.

Tests whether AI adoption announcements affect trading intensity and
volatility, whether AI-related CapEx amplifies those effects, and whether
the effects have attenuated over time (2015-2025) as AI became mainstream.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.event_study import compute_returns
from src.mock_data import data_loader

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLOR_VOLUME = "steelblue"
COLOR_VOLATILITY = "coral"
COLOR_CAPEX = "navy"
FIGURE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150

# ---------------------------------------------------------------------------
# 1A. Data loading & utilities
# ---------------------------------------------------------------------------


def assign_period_bin(date: pd.Timestamp) -> str:
    """Assign a date to one of four analysis period bins."""
    year = pd.Timestamp(date).year
    if year == 2022:
        return "2022"
    if year == 2023:
        return "2023"

    if year == 2024:
        return "2024"
    if year == 2025:
        return "2025"


def get_event_window(prices_df: pd.DataFrame, ticker: str,
                     event_date: pd.Timestamp,
                     pre_days: int = 30, post_days: int = 60) -> pd.DataFrame | None:
    """
    Extract price data around a single event.

    Returns DataFrame with a ``rel_day`` column (0 = event day).
    Handles weekends/holidays via searchsorted to snap to the
    next available trading day.
    """
    ticker_data = prices_df[prices_df["ticker"] == ticker].copy()
    if ticker_data.empty:
        return None

    ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)
    trading_dates = ticker_data["Date"].values

    idx = np.searchsorted(trading_dates, np.datetime64(pd.Timestamp(event_date)))
    if idx >= len(trading_dates):
        idx = len(trading_dates) - 1

    start = max(0, idx - pre_days)
    end = min(len(trading_dates) - 1, idx + post_days)

    if end - idx < 5:
        return None

    window = ticker_data.iloc[start:end + 1].copy()
    window["rel_day"] = np.arange(-(idx - start), end - idx + 1)
    return window


# ---------------------------------------------------------------------------
# 1B. Trading intensity (abnormal volume)
# ---------------------------------------------------------------------------

def compute_event_abnormal_volume_stats(
    prices_df: pd.DataFrame,
    events_df: pd.DataFrame,
    windows: list[tuple[int, int]] | None = None,
    pre_window: tuple[int, int] = (-30, -5),
):
    """
    Compute abnormal volume ratios for each event across multiple windows.

    Abnormal Volume = Volume_t / mean(Volume over pre_window).
    Returns one row per event with columns ``abvol_<start>_<end>`` for
    each window in *windows*.
    """
    if windows is None:
        windows = [(-1, 1), (0, 5), (0, 10), (0, 20)]

    records = []
    for i, row in events_df.iterrows():
        w = get_event_window(prices_df, row["ticker"],
                             row["announcement_date"], pre_days=40, post_days=25)
        if w is None:
            records.append({f"abvol_{s}_{e}": np.nan for s, e in windows})
            continue
        print(w["rel_day"])
        print(pre_window[0])
        pre_mask = (w["rel_day"] >= pre_window[0]) & (w["rel_day"] <= pre_window[1])
        pre_vol = w.loc[pre_mask, "Volume"].mean()
        if pre_vol is None or pre_vol == 0 or np.isnan(pre_vol):
            records.append({f"abvol_{s}_{e}": np.nan for s, e in windows})
            continue

        rec = {}
        for s, e in windows:
            mask = (w["rel_day"] >= s) & (w["rel_day"] <= e)
            post_vol = w.loc[mask, "Volume"].mean()
            rec[f"abvol_{s}_{e}"] = post_vol / pre_vol if not np.isnan(post_vol) else np.nan
        records.append(rec)

    result = pd.DataFrame(records, index=events_df.index)
    return pd.concat([events_df.reset_index(drop=True),
                      result.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# 1C. Volatility
# ---------------------------------------------------------------------------

def compute_max_drawdown(cumulative_returns: pd.Series) -> tuple[float, int]:
    """
    Compute maximum drawdown and days to trough from cumulative returns.

    Returns (max_drawdown, days_to_trough).
    """
    if cumulative_returns.empty:
        return 0.0, 0
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - running_max
    max_dd = drawdown.min()
    if max_dd == 0:
        return 0.0, 0
    trough_idx = drawdown.idxmin()
    # days_to_trough: position of the trough in the series
    days_to_trough = list(drawdown.index).index(trough_idx) + 1
    return abs(max_dd), days_to_trough


def compute_event_volatility_stats(
    prices_df: pd.DataFrame,
    spy_prices: pd.DataFrame,
    events_df: pd.DataFrame,
    pre_window: tuple[int, int] = (-30, -5),
    post_window: tuple[int, int] = (1, 30),
) -> pd.DataFrame:
    """
    Compute volatility statistics around each event.

    Returns DataFrame with columns: pre_vol, post_vol, vol_change,
    vol_ratio, max_drawdown, drawdown_speed.
    """
    stock_ret = compute_returns(prices_df, price_col="Adj_Close", group_col="ticker")
    spy_ret = compute_returns(spy_prices, price_col="Adj_Close", group_col=None)

    records = []
    for _, row in events_df.iterrows():
        ticker = row["ticker"]
        ann_date = pd.Timestamp(row["announcement_date"])

        w = get_event_window(stock_ret, ticker, ann_date, pre_days=40, post_days=35)
        if w is None or "log_return" not in w.columns:
            records.append(_empty_vol_record())
            continue

        # Align SPY returns
        spy_data = spy_ret.set_index("Date")["log_return"]
        w_indexed = w.set_index("Date")

        # Pre-event volatility
        pre_mask = (w["rel_day"].values >= pre_window[0]) & (w["rel_day"].values <= pre_window[1])
        pre_returns = w.loc[pre_mask, "log_return"].dropna()
        pre_vol = pre_returns.std() if len(pre_returns) > 2 else np.nan

        # Post-event volatility
        post_mask = (w["rel_day"].values >= post_window[0]) & (w["rel_day"].values <= post_window[1])
        post_returns = w.loc[post_mask, "log_return"].dropna()
        post_vol = post_returns.std() if len(post_returns) > 2 else np.nan

        # Abnormal returns for drawdown calculation
        post_dates = w.loc[post_mask, "Date"].values
        stock_r = w.loc[post_mask, "log_return"].values
        spy_r = spy_data.reindex(pd.DatetimeIndex(post_dates)).fillna(0).values
        ar = stock_r - spy_r
        car = pd.Series(np.cumsum(ar))

        max_dd, days_to_trough = compute_max_drawdown(car)
        dd_speed = max_dd / days_to_trough if days_to_trough > 0 else 0.0

        vol_change = post_vol - pre_vol if not (np.isnan(pre_vol) or np.isnan(post_vol)) else np.nan
        vol_ratio = post_vol / pre_vol if pre_vol and pre_vol > 0 and not np.isnan(pre_vol) else np.nan

        records.append({
            "pre_vol": pre_vol,
            "post_vol": post_vol,
            "vol_change": vol_change,
            "vol_ratio": vol_ratio,
            "max_drawdown": max_dd,
            "drawdown_speed": dd_speed,
        })

    vol_df = pd.DataFrame(records, index=events_df.index)
    return pd.concat([events_df.reset_index(drop=True),
                      vol_df.reset_index(drop=True)], axis=1)


def _empty_vol_record() -> dict:
    return {
        "pre_vol": np.nan, "post_vol": np.nan, "vol_change": np.nan,
        "vol_ratio": np.nan, "max_drawdown": np.nan, "drawdown_speed": np.nan,
    }


# ---------------------------------------------------------------------------
# 1D. CapEx matching
# ---------------------------------------------------------------------------

def match_capex_to_events(events_df: pd.DataFrame,
                          edgar_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (ticker, event_date), find the most recent EDGAR quarter
    where end_date < announcement_date (no look-ahead bias).

    Pivots concepts to columns and computes capex_intensity and rd_intensity.
    """
    edgar = edgar_df.copy()
    edgar["end_date"] = pd.to_datetime(edgar["end_date"])

    results = []
    for _, row in events_df.iterrows():
        ticker = row["ticker"]
        ann_date = pd.Timestamp(row["announcement_date"])

        # Filter: same ticker, before announcement
        mask = (edgar["ticker"] == ticker) & (edgar["end_date"] < ann_date)
        matched = edgar.loc[mask]

        if matched.empty:
            results.append({"capex_intensity": np.nan, "rd_intensity": np.nan})
            continue

        # Most recent quarter
        latest_date = matched["end_date"].max()
        latest = matched[matched["end_date"] == latest_date]

        # Pivot concepts
        capex_val = np.nan
        rd_val = np.nan
        for _, m in latest.iterrows():
            concept = m["concept"]
            if "Capital" in concept or "PaymentsToAcquire" in concept:
                capex_val = m["value"]
            elif "Research" in concept:
                rd_val = m["value"]

        # capex_intensity = CapEx level (we don't have PP&E net in mock, use raw)
        results.append({
            "capex_intensity": capex_val,
            "rd_intensity": rd_val,
        })

    capex_df = pd.DataFrame(results, index=events_df.index)
    return pd.concat([events_df.reset_index(drop=True),
                      capex_df.reset_index(drop=True)], axis=1)


def classify_capex_high_low(df: pd.DataFrame,
                            column: str = "capex_intensity") -> pd.DataFrame:
    """Add ``high_capex`` binary column via median split."""
    out = df.copy()
    median_val = out[column].median()
    out["high_capex"] = (out[column] >= median_val).astype(int)
    return out


# ---------------------------------------------------------------------------
# 1E. Statistical tests
# ---------------------------------------------------------------------------

def test_abnormal_volume_significance(vals: pd.Series | np.ndarray) -> dict:
    """
    One-sample t-test + Wilcoxon signed-rank (one-sided).
    H0: AbVol = 1  (no abnormal volume).
    """
    vals = pd.Series(vals).dropna()
    if len(vals) < 5:
        return {"t_stat": np.nan, "t_pval": np.nan,
                "w_stat": np.nan, "w_pval": np.nan, "n": len(vals)}
    t_stat, t_pval = stats.ttest_1samp(vals, 1.0)
    # One-sided: AbVol > 1
    t_pval_one = t_pval / 2 if t_stat > 0 else 1 - t_pval / 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w_stat, w_pval = stats.wilcoxon(vals - 1.0, alternative="greater")
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

    return {
        "t_stat": t_stat, "t_pval": t_pval_one,
        "w_stat": w_stat, "w_pval": w_pval,
        "n": len(vals), "mean": vals.mean(), "median": vals.median(),
    }


def test_paired_volatility_change(pre: pd.Series | np.ndarray,
                                  post: pd.Series | np.ndarray) -> dict:
    """
    Paired t-test + Wilcoxon signed-rank.
    H0: pre_vol = post_vol.
    """
    pre = pd.Series(pre)
    post = pd.Series(post)
    valid = pre.notna() & post.notna()
    pre, post = pre[valid], post[valid]

    if len(pre) < 5:
        return {"t_stat": np.nan, "t_pval": np.nan,
                "w_stat": np.nan, "w_pval": np.nan, "n": len(pre)}

    t_stat, t_pval = stats.ttest_rel(pre, post)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w_stat, w_pval = stats.wilcoxon(post - pre)
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

    return {
        "t_stat": t_stat, "t_pval": t_pval,
        "w_stat": w_stat, "w_pval": w_pval,
        "n": len(pre),
        "mean_diff": (post - pre).mean(),
    }


def test_period_differences(df: pd.DataFrame, value_col: str,
                            group_col: str = "period_bin") -> dict:
    """
    ANOVA + Kruskal-Wallis + pairwise Mann-Whitney (Bonferroni).
    H0: All period means / medians are equal.
    """
    groups = {name: grp[value_col].dropna()
              for name, grp in df.groupby(group_col) if len(grp[value_col].dropna()) > 0}

    if len(groups) < 2:
        return {"f_stat": np.nan, "f_pval": np.nan,
                "kw_stat": np.nan, "kw_pval": np.nan, "pairwise": {}}

    group_vals = list(groups.values())
    group_names = list(groups.keys())

    # ANOVA
    f_stat, f_pval = stats.f_oneway(*group_vals)

    # Kruskal-Wallis
    kw_stat, kw_pval = stats.kruskal(*group_vals)

    # Pairwise Mann-Whitney with Bonferroni
    n_comparisons = len(group_names) * (len(group_names) - 1) // 2
    pairwise = {}
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            u_stat, u_pval = stats.mannwhitneyu(
                group_vals[i], group_vals[j], alternative="two-sided"
            )
            pairwise[f"{group_names[i]} vs {group_names[j]}"] = {
                "u_stat": u_stat,
                "p_raw": u_pval,
                "p_bonferroni": min(u_pval * n_comparisons, 1.0),
            }

    return {
        "f_stat": f_stat, "f_pval": f_pval,
        "kw_stat": kw_stat, "kw_pval": kw_pval,
        "group_ns": {k: len(v) for k, v in groups.items()},
        "group_means": {k: v.mean() for k, v in groups.items()},
        "pairwise": pairwise,
    }


def test_capex_group_difference(high: pd.Series | np.ndarray,
                                low: pd.Series | np.ndarray) -> dict:
    """
    Two-sample t-test + Mann-Whitney U.
    H0: High CapEx group = Low CapEx group.
    """
    high = pd.Series(high).dropna()
    low = pd.Series(low).dropna()

    if len(high) < 3 or len(low) < 3:
        return {"t_stat": np.nan, "t_pval": np.nan,
                "u_stat": np.nan, "u_pval": np.nan,
                "n_high": len(high), "n_low": len(low)}

    t_stat, t_pval = stats.ttest_ind(high, low, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(high, low, alternative="two-sided")

    return {
        "t_stat": t_stat, "t_pval": t_pval,
        "u_stat": u_stat, "u_pval": u_pval,
        "n_high": len(high), "n_low": len(low),
        "mean_high": high.mean(), "mean_low": low.mean(),
    }


# ---------------------------------------------------------------------------
# 1F. Regression
# ---------------------------------------------------------------------------

def run_ols_regression(df: pd.DataFrame, dependent_var: str,
                       independent_vars: list[str],
                       add_period_dummies: bool = True) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression with optional period dummies.

    Returns statsmodels OLS result object.
    """
    working = df.dropna(subset=[dependent_var] + independent_vars).copy()

    X = working[independent_vars].copy()

    if add_period_dummies and "period_bin" in working.columns:
        dummies = pd.get_dummies(working["period_bin"], prefix="period", drop_first=True)
        # Convert boolean columns to int
        dummies = dummies.astype(int)
        X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)
    y = working[dependent_var]

    model = sm.OLS(y, X).fit()
    return model


# ---------------------------------------------------------------------------
# 1G. Visualization helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURE_DIR / name, dpi=DPI, bbox_inches="tight")


def plot_abnormal_volume_by_period(stats_df: pd.DataFrame,
                                   window_col: str = "abvol_0_5") -> plt.Figure:
    """Box + strip plot of abnormal volume by period bin."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = stats_df.dropna(subset=[window_col, "period_bin"])

    #sns.boxplot(data=data, x="period_bin", y=window_col,
                #color=COLOR_VOLUME, width=0.5, ax=ax, fliersize=0)
    #sns.stripplot(data=data, x="period_bin", y=window_col,
                  #color="black", alpha=0.3, size=3, ax=ax, jitter=True)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="No abnormal volume")
    ax.set_xlabel("Period")
    ax.set_ylabel(f"Abnormal Volume Ratio ({window_col})")
    ax.set_title("Abnormal Trading Volume by Period")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "stat_abvol_by_period.png")
    return fig


def plot_volatility_change_by_period(vol_df: pd.DataFrame) -> plt.Figure:
    """Box + strip plot of volatility change by period bin."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = vol_df.dropna(subset=["vol_change", "period_bin"])

    #sns.boxplot(data=data, x="period_bin", y="vol_change",
                #color=COLOR_VOLATILITY, width=0.5, ax=ax, fliersize=0)
    #sns.stripplot(data=data, x="period_bin", y="vol_change",
                  #color="black", alpha=0.3, size=3, ax=ax, jitter=True)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="No change")
    ax.set_xlabel("Period")
    ax.set_ylabel("Volatility Change (post - pre)")
    ax.set_title("Post-Event Volatility Change by Period")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "stat_vol_change_by_period.png")
    return fig


def plot_capex_interaction(df: pd.DataFrame,
                           reaction_col: str = "abvol_0_5") -> plt.Figure:
    """Grouped bar chart of market reaction by CapEx group and period."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data = df.dropna(subset=[reaction_col, "period_bin", "high_capex"])

    summary = data.groupby(["period_bin", "high_capex"])[reaction_col].agg(
        ["mean", "sem"]
    ).reset_index()
    summary["label"] = summary["high_capex"].map({1: "High CapEx", 0: "Low CapEx"})

    periods = sorted(summary["period_bin"].unique())
    x = np.arange(len(periods))
    width = 0.35

    for idx, (label, color) in enumerate([("High CapEx", COLOR_CAPEX),
                                           ("Low CapEx", COLOR_VOLUME)]):
        subset = summary[summary["label"] == label].set_index("period_bin").reindex(periods)
        ax.bar(x + idx * width, subset["mean"], width, label=label,
               color=color, alpha=0.8, yerr=subset["sem"], capsize=4)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(periods)
    ax.set_ylabel(f"Mean {reaction_col}")
    ax.set_title("Market Reaction by CapEx Intensity and Period")
    ax.legend()
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save_fig(fig, "stat_capex_interaction.png")
    return fig


def plot_time_trend_line(df: pd.DataFrame, y_col: str,
                         title: str | None = None) -> plt.Figure:
    """Line plot of mean y_col by period with 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = df.dropna(subset=[y_col, "period_bin"])
    summary = data.groupby("period_bin")[y_col].agg(["mean", "sem", "count"]).reset_index()
    summary["ci_95"] = summary["sem"] * 1.96

    ax.errorbar(summary["period_bin"], summary["mean"],
                yerr=summary["ci_95"], marker="o", capsize=5,
                color=COLOR_VOLUME, linewidth=2, markersize=8)

    for _, r in summary.iterrows():
        ax.annotate(f"n={int(r['count'])}", (r["period_bin"], r["mean"]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color="gray")

    ax.set_xlabel("Period")
    ax.set_ylabel(y_col)
    ax.set_title(title or f"Time Trend: {y_col}")
    fig.tight_layout()
    _save_fig(fig, f"stat_trend_{y_col}.png")
    return fig


def format_test_results_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Convert a dict of test results into a formatted summary table
    with significance stars.
    """
    rows = []
    for test_name, res in results.items():
        row = {"Test": test_name}
        for key, val in res.items():
            if key == "pairwise":
                continue
            if isinstance(val, float):
                if "pval" in key or key.startswith("p_"):
                    stars = _sig_stars(val)
                    row[key] = f"{val:.4f}{stars}"
                else:
                    row[key] = f"{val:.4f}"
            else:
                row[key] = val
        rows.append(row)
    return pd.DataFrame(rows).set_index("Test")


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return " ***"
    elif p < 0.01:
        return " **"
    elif p < 0.05:
        return " *"
    elif p < 0.1:
        return " +"
    return ""
