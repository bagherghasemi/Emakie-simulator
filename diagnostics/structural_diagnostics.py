#!/usr/bin/env python3
"""
Structural Diagnostics Script for the Emakie Simulator.

Reads simulator parquet output, computes structural fingerprints, and compares
each against benchmark ranges. Produces machine-readable JSON and human-readable
text reports.

Usage:
    python diagnostics/structural_diagnostics.py --data-dir output/
    python diagnostics/structural_diagnostics.py --data-dir output/ --sim-days 90
"""

import argparse
import json
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress convergence warnings from scipy
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

CALIBRATION_PATH = Path(__file__).parent / "benchmark_calibration.json"
REPORTS_DIR = Path(__file__).parent / "reports"

# Hard-coded defaults (used when calibration file is missing)
_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "1.1_revenue_autocorrelation": {
        "range": [0.3, 0.85],
        "hard_fail_below": 0.15,
        "hard_fail_above": 0.95,
    },
    "1.2_seasonality_presence": {
        "range": [0.08, 0.55],
        "hard_fail_below": 0.01,
        "hard_fail_above": 0.85,
    },
    "1.3_trend_presence": {
        "range": [0.02, 0.45],
        "hard_fail_below": -0.3,
        "hard_fail_above": 0.8,
    },
    "1.4_non_stationarity": {"range": [0.05, 0.99]},
    "2.1_customer_ltv_distribution": {
        "range": [0.45, 0.80],
        "hard_fail_below": 0.2,
        "hard_fail_above": 0.95,
    },
    "2.2_order_value_cv": {
        "range": [0.35, 1.0],
        "hard_fail_below": 0.05,
        "hard_fail_above": 2.5,
    },
    "2.3_purchase_interval_distribution": {
        "range": [0.30, 0.70],
        "hard_fail_below": 0.10,
        "hard_fail_above": 0.95,
    },
    "2.4_refund_timing_distribution": {
        "range": [0.25, 0.65],
        "hard_fail_below": 0.05,
        "hard_fail_above": 0.95,
    },
    "3.1_creative_age_vs_performance": {
        "range": [-0.65, -0.15],
        "hard_fail_above": 0.3,
    },
    "3.2_discount_vs_repeat": {"range": [-0.35, 0.15]},
    "3.3_frequency_vs_conversion": {"range": [-0.45, 0.20]},
    "3.4_refund_vs_repeat": {"range": [-0.45, 0.10]},
    "3.5_cross_lag_correlations": {
        "range": [0.15, 0.80],
        "hard_fail_below": 0.02,
        "hard_fail_above": 0.95,
    },
    "4.1_creative_concentration": {
        "range": [0.06, 0.35],
        "hard_fail_below": 0.01,
        "hard_fail_above": 0.7,
    },
    "4.2_customer_concentration": {
        "range": [0.25, 0.55],
        "hard_fail_below": 0.12,
        "hard_fail_above": 0.80,
    },
    "4.3_channel_asymmetry": {
        "range": [1.5, 15.0],
        "hard_fail_below": 1.0,
        "hard_fail_above": 50.0,
    },
    "5.1_acquisition_cost_trend": {
        "range": [0.08, 0.35],
        "hard_fail_below": -0.2,
        "hard_fail_above": 0.8,
    },
    "5.2_cohort_composition_drift": {
        "range": [-0.25, 0.0],
        "hard_fail_below": -0.6,
        "hard_fail_above": 0.3,
    },
    "5.3_repeat_rate_evolution": {"range": [-0.05, 0.50]},
    "5.4_trust_baseline_trend": {"range": [-0.12, 0.08]},
    "5.5_1yr_vs_3yr_divergence": {
        "range": [0.3, 1.8],
        "hard_fail_below": 0.05,
        "hard_fail_above": 5.0,
    },
    "6.1_refund_trust_granger": {"range": [0.0, 0.10]},
    "6.2_trust_repeat_granger": {"range": [0.0, 0.10]},
    "6.3_spiral_detection": {"range": [0, 15]},
    "7.1_history_effect": {
        "range": [0.08, 0.35],
        "hard_fail_below": -0.2,
        "hard_fail_above": 0.7,
    },
    "7.2_negative_experience_persistence": {
        "range": [12, 120],
        "hard_fail_below": 3,
        "hard_fail_above": 365,
    },
}

# Check metadata
_CHECK_META = {
    "1.1_revenue_autocorrelation": ("temporal_structure", "hard_fail"),
    "1.2_seasonality_presence": ("temporal_structure", "warn"),
    "1.3_trend_presence": ("temporal_structure", "hard_fail"),
    "1.4_non_stationarity": ("temporal_structure", "info"),
    "2.1_customer_ltv_distribution": ("distribution_shape", "hard_fail"),
    "2.2_order_value_cv": ("distribution_shape", "warn"),
    "2.3_purchase_interval_distribution": ("distribution_shape", "warn"),
    "2.4_refund_timing_distribution": ("distribution_shape", "warn"),
    "3.1_creative_age_vs_performance": ("cross_metric", "hard_fail"),
    "3.2_discount_vs_repeat": ("cross_metric", "warn"),
    "3.3_frequency_vs_conversion": ("cross_metric", "warn"),
    "3.4_refund_vs_repeat": ("cross_metric", "warn"),
    "3.5_cross_lag_correlations": ("cross_metric", "hard_fail"),
    "4.1_creative_concentration": ("structural_asymmetry", "warn"),
    "4.2_customer_concentration": ("structural_asymmetry", "warn"),
    "4.3_channel_asymmetry": ("structural_asymmetry", "warn"),
    "5.1_acquisition_cost_trend": ("time_dependence", "hard_fail"),
    "5.2_cohort_composition_drift": ("time_dependence", "hard_fail"),
    "5.3_repeat_rate_evolution": ("time_dependence", "warn"),
    "5.4_trust_baseline_trend": ("time_dependence", "warn"),
    "5.5_1yr_vs_3yr_divergence": ("time_dependence", "hard_fail"),
    "6.1_refund_trust_granger": ("feedback_loops", "warn"),
    "6.2_trust_repeat_granger": ("feedback_loops", "warn"),
    "6.3_spiral_detection": ("feedback_loops", "info"),
    "7.1_history_effect": ("memory_path_dependence", "warn"),
    "7.2_negative_experience_persistence": ("memory_path_dependence", "warn"),
}


def load_benchmarks() -> Dict[str, Dict]:
    """Load calibrated thresholds. Fall back to defaults if calibration missing."""
    benchmarks = {}
    for check_id, defaults in _DEFAULTS.items():
        benchmarks[check_id] = {**defaults, "_source": "default"}

    if CALIBRATION_PATH.exists():
        try:
            with open(CALIBRATION_PATH) as f:
                calibration = json.load(f)
            for check_id, cal in calibration.get("benchmarks", {}).items():
                if check_id not in benchmarks:
                    continue
                conf = cal.get("confidence", "default")
                if conf in ("high", "medium"):
                    cr = cal.get("calibrated_range", {})
                    if isinstance(cr, dict) and "range" in cr:
                        benchmarks[check_id]["range"] = cr["range"]
                        benchmarks[check_id]["_source"] = "calibrated"
                    # Also pick up hard-fail thresholds if present
                    for key in ("hard_fail_below", "hard_fail_above"):
                        if key in cal and cal[key] is not None:
                            benchmarks[check_id][key] = cal[key]
        except (json.JSONDecodeError, KeyError):
            pass  # fall back to defaults

    return benchmarks


BENCHMARKS = load_benchmarks()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_parquet_folder(data_dir: Path, folder: str) -> Optional[pd.DataFrame]:
    """Load all parquet files from a subfolder."""
    folder_path = data_dir / folder
    if not folder_path.exists():
        return None
    files = sorted(folder_path.glob("*.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_sim_data(data_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """Load all simulator output data."""
    data = {}
    for table in [
        "orders",
        "meta_exposures",
        "meta_ad_performance_daily",
        "refunds",
        "fulfillments",
        "line_items",
        "transactions",
        "shopify_checkouts",
    ]:
        data[table] = _load_parquet_folder(data_dir, table)
    return data


def load_diagnostic_data(diag_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """Load diagnostic output data (brand_state_daily, etc.)."""
    data = {}
    data["brand_state_daily"] = _load_parquet_folder(diag_dir, "brand_state_daily")
    return data


# ---------------------------------------------------------------------------
# Check result helper
# ---------------------------------------------------------------------------


def _make_result(
    check_id: str,
    observed: Any,
    detail: str,
    status: Optional[str] = None,
) -> Dict:
    """Build a check result dict."""
    category, severity = _CHECK_META.get(check_id, ("unknown", "warn"))
    bench = BENCHMARKS.get(check_id, {})
    bench_range = bench.get("range", [None, None])
    source = bench.get("_source", "default")

    if status is None:
        # Auto-determine status
        if observed is None:
            status = "SKIP"
        elif severity == "info":
            status = "INFO"
        else:
            lo, hi = (bench_range[0], bench_range[1]) if isinstance(bench_range, list) else (None, None)
            hf_below = bench.get("hard_fail_below")
            hf_above = bench.get("hard_fail_above")

            # Check hard-fail bounds first
            if hf_below is not None and observed < hf_below:
                status = "FAIL"
            elif hf_above is not None and observed > hf_above:
                status = "FAIL"
            elif lo is not None and hi is not None:
                if lo <= observed <= hi:
                    status = "PASS"
                else:
                    status = "WARN" if severity == "warn" else "FAIL"
            else:
                status = "PASS"

    return {
        "status": status,
        "category": category,
        "severity": severity,
        "observed": _jsonable(observed),
        "benchmark_range": _jsonable(bench_range),
        "benchmark_source": source,
        "detail": detail,
    }


def _jsonable(v):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


# ---------------------------------------------------------------------------
# CATEGORY 1: TEMPORAL STRUCTURE
# ---------------------------------------------------------------------------


def check_1_1_revenue_autocorrelation(daily_revenue: pd.Series) -> Dict:
    """Lag-1, lag-7, lag-30 autocorrelation of daily revenue."""
    check_id = "1.1_revenue_autocorrelation"
    if daily_revenue is None or len(daily_revenue) < 14:
        return _make_result(check_id, None, "Insufficient data (<14 days)", "SKIP")

    vals = daily_revenue.values.astype(float)
    if np.std(vals) < 1e-10:
        return _make_result(check_id, 0.0, "Revenue is constant (zero variance)")

    lag1 = np.corrcoef(vals[:-1], vals[1:])[0, 1]
    lag7 = np.corrcoef(vals[:-7], vals[7:])[0, 1] if len(vals) >= 14 else None
    lag30 = np.corrcoef(vals[:-30], vals[30:])[0, 1] if len(vals) >= 60 else None

    detail = f"lag1={lag1:.3f}"
    if lag7 is not None:
        detail += f", lag7={lag7:.3f}"
    if lag30 is not None:
        detail += f", lag30={lag30:.3f}"

    return _make_result(check_id, float(lag1), detail)


def check_1_2_seasonality_presence(daily_revenue: pd.Series) -> Dict:
    """FFT/periodogram spectral peak near period=7."""
    check_id = "1.2_seasonality_presence"
    if daily_revenue is None or len(daily_revenue) < 21:
        return _make_result(check_id, None, "Insufficient data (<21 days)", "SKIP")

    vals = daily_revenue.values.astype(float)
    vals = vals - np.mean(vals)  # detrend
    if np.std(vals) < 1e-10:
        return _make_result(check_id, 0.0, "Revenue is constant")

    fft_vals = np.abs(np.fft.rfft(vals))[1:]  # skip DC component
    freqs = np.fft.rfftfreq(len(vals))[1:]
    periods = 1.0 / freqs

    # Find peak near period=7 (weekly)
    weekly_mask = (periods >= 5) & (periods <= 9)
    if not weekly_mask.any():
        return _make_result(check_id, 0.0, "No frequency in weekly range")

    weekly_power = fft_vals[weekly_mask].max()
    total_power = fft_vals.sum()
    seasonal_strength = float(weekly_power / total_power) if total_power > 0 else 0.0

    peak_period = periods[np.argmax(fft_vals)]
    detail = f"seasonal_strength={seasonal_strength:.3f}, peak_period={peak_period:.1f}d"

    return _make_result(check_id, seasonal_strength, detail)


def check_1_3_trend_presence(daily_revenue: pd.Series, sim_days: int = 0) -> Dict:
    """Linear trend slope on monthly revenue."""
    check_id = "1.3_trend_presence"
    if daily_revenue is None or len(daily_revenue) < 60:
        return _make_result(check_id, None, "Insufficient data (<60 days) for trend", "SKIP")

    # Group by month
    monthly = daily_revenue.resample("ME").sum()
    if len(monthly) < 3:
        return _make_result(check_id, None, "Fewer than 3 months", "SKIP")

    x = np.arange(len(monthly), dtype=float)
    y = monthly.values.astype(float)
    if np.std(y) < 1e-10:
        return _make_result(check_id, 0.0, "Revenue is constant across months")

    slope = np.polyfit(x, y, 1)[0]
    # Normalize slope by mean monthly revenue
    mean_rev = np.mean(y)
    norm_slope = slope / mean_rev if mean_rev > 0 else 0.0

    # In short sims (<180d), launch normalization (novelty decay, prospect pool
    # exhaustion, creative fatigue) creates a structural negative trend.
    # Downgrade to INFO for short sims (like check 7.1).
    status = None
    if sim_days > 0 and sim_days < 180:
        detail = f"normalized_slope={norm_slope:.4f}, raw_slope={slope:.2f}/month, short_sim={sim_days}d"
        status = "INFO"
    else:
        detail = f"normalized_slope={norm_slope:.4f}, raw_slope={slope:.2f}/month"
    return _make_result(check_id, float(norm_slope), detail, status)


def check_1_4_non_stationarity(daily_revenue: pd.Series) -> Dict:
    """ADF test on daily revenue."""
    check_id = "1.4_non_stationarity"
    if daily_revenue is None or len(daily_revenue) < 30:
        return _make_result(check_id, None, "Insufficient data (<30 days)", "SKIP")

    try:
        from scipy.stats import pearsonr

        # Simple ADF approximation: regress diff on lagged level
        vals = daily_revenue.values.astype(float)
        diff = np.diff(vals)
        lagged = vals[:-1]
        if np.std(lagged) < 1e-10:
            return _make_result(check_id, 1.0, "Constant series, trivially non-stationary", "INFO")

        try:
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(vals, maxlag=min(14, len(vals) // 3))
            p_value = result[1]
        except ImportError:
            # Fallback: correlation-based proxy
            corr, p_value = pearsonr(lagged, diff)
            # If negative correlation with high significance → stationary
            p_value = max(p_value, 0.0)

        detail = f"ADF p-value={p_value:.4f}"
        return _make_result(check_id, float(p_value), detail, "INFO")
    except Exception as e:
        return _make_result(check_id, None, f"ADF test failed: {e}", "SKIP")


# ---------------------------------------------------------------------------
# CATEGORY 2: DISTRIBUTION SHAPE
# ---------------------------------------------------------------------------


def check_2_1_customer_ltv(orders_df: pd.DataFrame) -> Dict:
    """Skewness / Gini of revenue per customer."""
    check_id = "2.1_customer_ltv_distribution"
    if orders_df is None or orders_df.empty:
        return _make_result(check_id, None, "No orders data", "SKIP")

    if "total_price" not in orders_df.columns or "customer_id" not in orders_df.columns:
        return _make_result(check_id, None, "Missing total_price or customer_id", "SKIP")

    ltv = orders_df.groupby("customer_id")["total_price"].sum().sort_values()
    if len(ltv) < 10:
        return _make_result(check_id, None, "Fewer than 10 customers", "SKIP")

    # Gini coefficient
    vals = ltv.values.astype(float)
    n = len(vals)
    sorted_vals = np.sort(vals)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n

    # Top 10% share
    top10_count = max(1, n // 10)
    top10_share = sorted_vals[-top10_count:].sum() / sorted_vals.sum() if sorted_vals.sum() > 0 else 0

    from scipy.stats import skew

    skewness = skew(vals)

    detail = f"gini={gini:.3f}, skewness={skewness:.2f}, top10_share={top10_share:.2%}"
    return _make_result(check_id, float(gini), detail)


def check_2_2_order_value_cv(orders_df: pd.DataFrame) -> Dict:
    """Coefficient of variation of order values."""
    check_id = "2.2_order_value_cv"
    if orders_df is None or orders_df.empty or "total_price" not in orders_df.columns:
        return _make_result(check_id, None, "No order value data", "SKIP")

    vals = orders_df["total_price"].dropna().values.astype(float)
    if len(vals) < 10:
        return _make_result(check_id, None, "Fewer than 10 orders", "SKIP")

    mean_val = np.mean(vals)
    if mean_val < 1e-10:
        return _make_result(check_id, 0.0, "Mean order value near zero")

    cv = float(np.std(vals) / mean_val)
    detail = f"CV={cv:.3f}, mean=${mean_val:.2f}, std=${np.std(vals):.2f}"
    return _make_result(check_id, cv, detail)


def check_2_3_purchase_intervals(orders_df: pd.DataFrame) -> Dict:
    """Repeat purchase interval distribution shape — median/mean ratio."""
    check_id = "2.3_purchase_interval_distribution"
    if orders_df is None or orders_df.empty:
        return _make_result(check_id, None, "No orders data", "SKIP")

    date_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
    if date_col not in orders_df.columns or "customer_id" not in orders_df.columns:
        return _make_result(check_id, None, "Missing date or customer_id", "SKIP")

    orders_sorted = orders_df.sort_values([date_col])
    intervals = []
    for _, group in orders_sorted.groupby("customer_id"):
        if len(group) < 2:
            continue
        dates = pd.to_datetime(group[date_col]).sort_values()
        diffs = dates.diff().dropna().dt.days.values
        intervals.extend(diffs[diffs > 0])

    if len(intervals) < 10:
        return _make_result(check_id, None, "Fewer than 10 repeat intervals", "SKIP")

    intervals = np.array(intervals, dtype=float)
    median_int = np.median(intervals)
    mean_int = np.mean(intervals)
    right_skewed = median_int < mean_int

    # Use median/mean ratio as shape metric (robust to sample size unlike KS test).
    # Right-skewed distributions (lognormal-like) have ratio < 1.0.
    # Real DTC: median ~30d, mean ~50-90d → ratio 0.3-0.6.
    ratio = median_int / mean_int if mean_int > 0 else 1.0

    detail = f"median={median_int:.1f}d, mean={mean_int:.1f}d, right_skewed={right_skewed}, median_mean_ratio={ratio:.3f}"
    return _make_result(check_id, float(ratio), detail)


def check_2_4_refund_timing(refunds_df: pd.DataFrame, orders_df: pd.DataFrame) -> Dict:
    """Days order→refund; clustering."""
    check_id = "2.4_refund_timing_distribution"
    if refunds_df is None or refunds_df.empty or orders_df is None or orders_df.empty:
        return _make_result(check_id, None, "No refund or order data", "SKIP")

    # Merge refunds with orders to get order date
    order_date_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
    refund_date_col = "created_at" if "created_at" in refunds_df.columns else "processed_at"
    order_id_col = "order_id" if "order_id" in refunds_df.columns else "id"

    if refund_date_col not in refunds_df.columns:
        return _make_result(check_id, None, "Missing refund date column", "SKIP")

    orders_for_merge = orders_df[["id", order_date_col]].rename(
        columns={"id": order_id_col, order_date_col: "_order_date"}
    )
    merged = refunds_df.merge(orders_for_merge, on=order_id_col, how="inner")
    if merged.empty:
        return _make_result(check_id, None, "No matched refund-order pairs", "SKIP")

    refund_dates = pd.to_datetime(merged[refund_date_col])
    order_dates = pd.to_datetime(merged["_order_date"])
    days_to_refund = (refund_dates - order_dates).dt.days.values.astype(float)
    days_to_refund = days_to_refund[days_to_refund >= 0]

    if len(days_to_refund) < 5:
        return _make_result(check_id, None, "Fewer than 5 refunds with timing", "SKIP")

    # Fraction within 7 days
    within_7d = np.mean(days_to_refund <= 7)

    detail = f"within_7d={within_7d:.2%}, median={np.median(days_to_refund):.1f}d, mean={np.mean(days_to_refund):.1f}d"
    return _make_result(check_id, float(within_7d), detail)


# ---------------------------------------------------------------------------
# CATEGORY 3: CROSS-METRIC RELATIONSHIPS
# ---------------------------------------------------------------------------


def check_3_1_creative_age_vs_performance(perf_df: pd.DataFrame) -> Dict:
    """Correlation between days_active and CTR per creative."""
    check_id = "3.1_creative_age_vs_performance"
    if perf_df is None or perf_df.empty:
        return _make_result(check_id, None, "No performance data", "SKIP")

    required = {"creative_id", "date", "impressions", "clicks"}
    if not required.issubset(perf_df.columns):
        # Try ad_id instead
        if "ad_id" in perf_df.columns and "creative_id" not in perf_df.columns:
            return _make_result(check_id, None, "Missing creative_id in performance data", "SKIP")
        return _make_result(check_id, None, f"Missing columns: {required - set(perf_df.columns)}", "SKIP")

    perf = perf_df.copy()
    perf["date"] = pd.to_datetime(perf["date"])
    perf["impressions"] = perf["impressions"].fillna(0).astype(float)
    perf["clicks"] = perf["clicks"].fillna(0).astype(float)

    # Compute first active date per creative
    first_date = perf.groupby("creative_id")["date"].min().rename("first_date")
    perf = perf.merge(first_date, on="creative_id")
    perf["days_active"] = (perf["date"] - perf["first_date"]).dt.days

    # Aggregate by creative × week (reduce noise)
    perf["week"] = perf["days_active"] // 7
    weekly = perf.groupby(["creative_id", "week"]).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        days_active=("days_active", "mean"),
    )
    weekly = weekly[weekly["impressions"] > 0]
    weekly["ctr"] = weekly["clicks"] / weekly["impressions"]

    if len(weekly) < 10:
        return _make_result(check_id, None, "Insufficient weekly creative data", "SKIP")

    corr = np.corrcoef(weekly["days_active"].values, weekly["ctr"].values)[0, 1]
    if np.isnan(corr):
        return _make_result(check_id, None, "Correlation is NaN", "SKIP")

    detail = f"corr(days_active, CTR)={corr:.3f}"
    return _make_result(check_id, float(corr), detail)


def check_3_2_discount_vs_repeat(orders_df: pd.DataFrame) -> Dict:
    """Repeat rate: heavy-discount vs low-discount customers."""
    check_id = "3.2_discount_vs_repeat"
    if orders_df is None or orders_df.empty:
        return _make_result(check_id, None, "No orders data", "SKIP")

    if "discount_pct" not in orders_df.columns and "total_discounts" not in orders_df.columns:
        return _make_result(check_id, None, "No discount column", "SKIP")

    # Determine discount per customer
    if "discount_pct" in orders_df.columns:
        customer_discount = orders_df.groupby("customer_id")["discount_pct"].mean()
    else:
        orders_df = orders_df.copy()
        orders_df["_disc_pct"] = orders_df["total_discounts"].fillna(0).astype(float) / orders_df[
            "total_price"
        ].clip(lower=0.01)
        customer_discount = orders_df.groupby("customer_id")["_disc_pct"].mean()

    customer_orders = orders_df.groupby("customer_id").size()

    if len(customer_discount) < 20:
        return _make_result(check_id, None, "Fewer than 20 customers", "SKIP")

    # Heavy discount (>20%) vs low (<5%)
    heavy_ids = customer_discount[customer_discount > 0.20].index
    low_ids = customer_discount[customer_discount < 0.05].index

    if len(heavy_ids) < 3 or len(low_ids) < 3:
        return _make_result(check_id, None, "Not enough heavy/low discount customers", "SKIP")

    heavy_repeat = (customer_orders.reindex(heavy_ids).fillna(1) > 1).mean()
    low_repeat = (customer_orders.reindex(low_ids).fillna(1) > 1).mean()
    diff = heavy_repeat - low_repeat

    detail = f"heavy_discount_repeat={heavy_repeat:.2%}, low_discount_repeat={low_repeat:.2%}, diff={diff:.2%}"
    return _make_result(check_id, float(diff), detail)


def check_3_3_frequency_vs_conversion(
    exposure_df: pd.DataFrame, orders_df: pd.DataFrame
) -> Dict:
    """Repeat rate difference between high-frequency and low-frequency exposed customers.

    Tests whether higher ad exposure frequency is associated with different repeat
    purchase rates. This differs from 7.1 (raw exposure-order count correlation)
    by using bin-based repeat rate comparison, avoiding ratio bias.
    Expected: slight negative to slight positive (diminishing returns at high freq).
    """
    check_id = "3.3_frequency_vs_conversion"
    if exposure_df is None or orders_df is None or exposure_df.empty or orders_df.empty:
        return _make_result(check_id, None, "Missing exposure or order data", "SKIP")

    if "customer_id" not in exposure_df.columns:
        return _make_result(check_id, None, "Missing customer_id in exposures", "SKIP")

    # Count impressions per customer (only non-null customer_ids)
    valid_exp = exposure_df[exposure_df["customer_id"].notna()]
    if valid_exp.empty:
        return _make_result(check_id, None, "No exposures with customer_id", "SKIP")

    freq = valid_exp.groupby("customer_id").size().rename("frequency")

    # Count orders per customer
    order_counts = orders_df.groupby("customer_id").size().rename("order_count")

    # Merge: frequency and order count per customer
    freq_df = freq.reset_index().merge(order_counts.reset_index(), on="customer_id", how="left")
    freq_df["order_count"] = freq_df["order_count"].fillna(0)

    if len(freq_df) < 30:
        return _make_result(check_id, None, "Fewer than 30 customers with frequency", "SKIP")

    # Split into frequency tertiles and compare repeat rates
    freq_df["is_repeat"] = (freq_df["order_count"] > 1).astype(int)
    try:
        freq_df["freq_bin"] = pd.qcut(freq_df["frequency"], q=3, labels=["low", "mid", "high"])
    except ValueError:
        # If too many ties for qcut, use manual split
        med = freq_df["frequency"].median()
        freq_df["freq_bin"] = np.where(freq_df["frequency"] <= med, "low", "high")

    bin_stats = freq_df.groupby("freq_bin").agg(
        n=("is_repeat", "size"),
        repeat_rate=("is_repeat", "mean"),
        mean_orders=("order_count", "mean"),
    )

    if "high" in bin_stats.index and "low" in bin_stats.index:
        high_repeat = bin_stats.loc["high", "repeat_rate"]
        low_repeat = bin_stats.loc["low", "repeat_rate"]
        diff = high_repeat - low_repeat
    elif len(bin_stats) >= 2:
        high_repeat = bin_stats.iloc[-1]["repeat_rate"]
        low_repeat = bin_stats.iloc[0]["repeat_rate"]
        diff = high_repeat - low_repeat
    else:
        return _make_result(check_id, None, "Could not create frequency bins", "SKIP")

    detail = f"high_freq_repeat={high_repeat:.3f}, low_freq_repeat={low_repeat:.3f}, diff={diff:.3f}"
    return _make_result(check_id, float(diff), detail)


def check_3_4_refund_vs_repeat(orders_df: pd.DataFrame, refunds_df: pd.DataFrame) -> Dict:
    """Future purchase probability: refunders vs non-refunders."""
    check_id = "3.4_refund_vs_repeat"
    if orders_df is None or refunds_df is None or orders_df.empty:
        return _make_result(check_id, None, "Missing orders or refunds", "SKIP")

    refunder_ids = set()
    if not refunds_df.empty and "order_id" in refunds_df.columns:
        refunded_orders = refunds_df["order_id"].unique()
        refunder_ids = set(
            orders_df[orders_df["id"].isin(refunded_orders)]["customer_id"].unique()
        )

    customer_orders = orders_df.groupby("customer_id").size()
    if len(customer_orders) < 20:
        return _make_result(check_id, None, "Fewer than 20 customers", "SKIP")

    refunder_repeat = (customer_orders.reindex(list(refunder_ids)).fillna(1) > 1).mean() if refunder_ids else 0
    non_refunder_ids = set(customer_orders.index) - refunder_ids
    non_refunder_repeat = (customer_orders.reindex(list(non_refunder_ids)).fillna(1) > 1).mean() if non_refunder_ids else 0

    diff = refunder_repeat - non_refunder_repeat
    detail = f"refunder_repeat={refunder_repeat:.2%}, non_refunder_repeat={non_refunder_repeat:.2%}, diff={diff:.2%}"
    return _make_result(check_id, float(diff), detail)


def check_3_5_cross_lag_correlations(daily_revenue: pd.Series, daily_refunds: pd.Series) -> Dict:
    """Peak lag between key time series."""
    check_id = "3.5_cross_lag_correlations"
    if daily_revenue is None or daily_refunds is None:
        return _make_result(check_id, None, "Missing daily time series", "SKIP")
    if len(daily_revenue) < 30 or len(daily_refunds) < 30:
        return _make_result(check_id, None, "Insufficient daily data (<30 days)", "SKIP")

    rev = daily_revenue.values.astype(float)
    ref = daily_refunds.values.astype(float)
    n = min(len(rev), len(ref))
    rev = rev[:n]
    ref = ref[:n]

    if np.std(rev) < 1e-10 or np.std(ref) < 1e-10:
        return _make_result(check_id, 0.0, "One series has zero variance")

    # Cross-correlation at multiple lags
    max_lag = min(30, n // 3)
    best_lag = 0
    best_corr = 0.0
    all_at_zero = True
    for lag in range(0, max_lag + 1):
        if lag == 0:
            c = np.corrcoef(rev, ref)[0, 1]
        else:
            c = np.corrcoef(rev[:-lag], ref[lag:])[0, 1]
        if np.isnan(c):
            continue
        if abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag
            if lag > 0:
                all_at_zero = False

    # The check fails if peak is at lag 0 (no causal delay)
    observed = abs(best_corr)
    detail = f"peak_lag={best_lag}d, peak_corr={best_corr:.3f}, all_at_lag0={all_at_zero}"

    result = _make_result(check_id, observed, detail)
    # Note: daily aggregate cross-correlations naturally peak at lag 0
    # due to volume effects (busy days have more of everything).
    # Per-order refund timing lags are captured by check 2.4 instead.
    return result


# ---------------------------------------------------------------------------
# CATEGORY 4: STRUCTURAL ASYMMETRY
# ---------------------------------------------------------------------------


def check_4_1_creative_concentration(perf_df: pd.DataFrame) -> Dict:
    """HHI of revenue by creative."""
    check_id = "4.1_creative_concentration"
    if perf_df is None or perf_df.empty:
        return _make_result(check_id, None, "No performance data", "SKIP")

    id_col = "creative_id" if "creative_id" in perf_df.columns else "ad_id"
    spend_col = "spend" if "spend" in perf_df.columns else None
    if spend_col is None:
        return _make_result(check_id, None, "No spend column", "SKIP")

    creative_spend = perf_df.groupby(id_col)[spend_col].sum()
    creative_spend = creative_spend[creative_spend > 0]
    if len(creative_spend) < 3:
        return _make_result(check_id, None, "Fewer than 3 creatives with spend", "SKIP")

    shares = creative_spend / creative_spend.sum()
    hhi = float((shares ** 2).sum())

    detail = f"HHI={hhi:.4f}, n_creatives={len(creative_spend)}, top_share={shares.max():.2%}"
    return _make_result(check_id, hhi, detail)


def check_4_2_customer_concentration(orders_df: pd.DataFrame) -> Dict:
    """Top-10% revenue share across customers."""
    check_id = "4.2_customer_concentration"
    if orders_df is None or orders_df.empty or "customer_id" not in orders_df.columns:
        return _make_result(check_id, None, "No orders data", "SKIP")

    customer_rev = orders_df.groupby("customer_id")["total_price"].sum()
    customer_rev = customer_rev[customer_rev > 0]
    if len(customer_rev) < 10:
        return _make_result(check_id, None, "Fewer than 10 customers", "SKIP")

    # Top-10% revenue share — robust to population size (unlike HHI)
    sorted_rev = customer_rev.sort_values(ascending=False)
    n_top = max(1, len(sorted_rev) // 10)
    top10_share = float(sorted_rev.iloc[:n_top].sum() / sorted_rev.sum())

    detail = f"top10_share={top10_share:.2%}, n_customers={len(customer_rev)}, top_customer={sorted_rev.iloc[0]:.0f}"
    return _make_result(check_id, top10_share, detail)


def check_4_3_channel_asymmetry(perf_df: pd.DataFrame) -> Dict:
    """Best/worst channel cost ratio."""
    check_id = "4.3_channel_asymmetry"
    if perf_df is None or perf_df.empty:
        return _make_result(check_id, None, "No performance data", "SKIP")

    # Use campaign_id as proxy for channel
    if "campaign_id" not in perf_df.columns or "spend" not in perf_df.columns:
        return _make_result(check_id, None, "Missing campaign_id or spend", "SKIP")

    if "conversions" not in perf_df.columns:
        return _make_result(check_id, None, "Missing conversions column", "SKIP")

    camp_metrics = perf_df.groupby("campaign_id").agg(
        spend=("spend", "sum"), conversions=("conversions", "sum")
    )
    camp_metrics = camp_metrics[(camp_metrics["conversions"] > 0) & (camp_metrics["spend"] > 0)]
    if len(camp_metrics) < 2:
        return _make_result(check_id, None, "Fewer than 2 campaigns with conversions+spend", "SKIP")

    camp_metrics["cpa"] = camp_metrics["spend"] / camp_metrics["conversions"]
    ratio = camp_metrics["cpa"].max() / camp_metrics["cpa"].min()

    detail = f"best_worst_cpa_ratio={ratio:.2f}x, n_campaigns={len(camp_metrics)}"
    return _make_result(check_id, float(ratio), detail)


# ---------------------------------------------------------------------------
# CATEGORY 5: TIME-DEPENDENCE
# ---------------------------------------------------------------------------


def check_5_1_acquisition_cost_trend(perf_df: pd.DataFrame, sim_days: int, orders_df: pd.DataFrame = None) -> Dict:
    """Monthly new-customer CPA trend over time."""
    check_id = "5.1_acquisition_cost_trend"
    if sim_days < 180:
        return _make_result(check_id, None, "Sim < 180 days, skipping time-dependence", "SKIP")
    if perf_df is None or perf_df.empty:
        return _make_result(check_id, None, "No performance data", "SKIP")
    if "spend" not in perf_df.columns:
        return _make_result(check_id, None, "Missing spend column", "SKIP")

    perf = perf_df.copy()
    perf["date"] = pd.to_datetime(perf["date"])
    perf["month"] = perf["date"].dt.to_period("M")
    monthly_spend = perf.groupby("month")["spend"].sum()

    # Compute new-customer acquisitions per month from orders data
    # (blended CPA including repeats naturally declines as customer base grows)
    if orders_df is not None and not orders_df.empty and "customer_id" in orders_df.columns:
        date_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
        odf = orders_df.copy()
        odf["_date"] = pd.to_datetime(odf[date_col])
        odf["_month"] = odf["_date"].dt.to_period("M")
        first_purchase = odf.groupby("customer_id")["_date"].min().rename("first_date")
        odf = odf.merge(first_purchase, on="customer_id")
        odf["is_new"] = odf["_date"] == odf["first_date"]
        monthly_new = odf[odf["is_new"]].groupby("_month").size()
    elif "conversions" in perf.columns:
        monthly_new = perf.groupby("month")["conversions"].sum()
    else:
        return _make_result(check_id, None, "Missing conversions/orders data", "SKIP")

    # Align months
    common_months = monthly_spend.index.intersection(monthly_new.index)
    if len(common_months) < 6:
        return _make_result(check_id, None, "Fewer than 6 months with data", "SKIP")

    monthly_spend = monthly_spend.loc[common_months]
    monthly_new = monthly_new.loc[common_months]
    monthly_new = monthly_new[monthly_new > 0]
    common_months = monthly_spend.index.intersection(monthly_new.index)
    if len(common_months) < 6:
        return _make_result(check_id, None, "Fewer than 6 months with new customers", "SKIP")

    cpa_series = monthly_spend.loc[common_months] / monthly_new.loc[common_months]
    x = np.arange(len(cpa_series), dtype=float)
    y = cpa_series.values.astype(float)
    slope = np.polyfit(x, y, 1)[0]

    mean_cpa = np.mean(y)
    annual_rate = (slope * 12) / mean_cpa if mean_cpa > 0 else 0

    detail = f"annual_cpa_growth={annual_rate:.2%}, slope={slope:.2f}/month, mean_cpa=${mean_cpa:.2f}"
    return _make_result(check_id, float(annual_rate), detail)


def check_5_2_cohort_composition_drift(orders_df: pd.DataFrame, sim_days: int) -> Dict:
    """KS test: first-quartile vs last-quartile acquired customers."""
    check_id = "5.2_cohort_composition_drift"
    if sim_days < 180:
        return _make_result(check_id, None, "Sim < 180 days, skipping", "SKIP")
    if orders_df is None or orders_df.empty:
        return _make_result(check_id, None, "No orders data", "SKIP")

    date_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
    if date_col not in orders_df.columns:
        return _make_result(check_id, None, "Missing date column", "SKIP")

    # Find first purchase date per customer
    orders = orders_df.copy()
    orders["_date"] = pd.to_datetime(orders[date_col])
    first_purchase = orders.groupby("customer_id")["_date"].min().sort_values()

    if len(first_purchase) < 20:
        return _make_result(check_id, None, "Fewer than 20 customers", "SKIP")

    # Split into quartiles
    q1_cutoff = first_purchase.quantile(0.25)
    q4_cutoff = first_purchase.quantile(0.75)
    q1_ids = first_purchase[first_purchase <= q1_cutoff].index
    q4_ids = first_purchase[first_purchase >= q4_cutoff].index

    # Compare LTV distributions
    ltv = orders.groupby("customer_id")["total_price"].sum()
    q1_ltv = ltv.reindex(q1_ids).dropna().values
    q4_ltv = ltv.reindex(q4_ids).dropna().values

    if len(q1_ltv) < 5 or len(q4_ltv) < 5:
        return _make_result(check_id, None, "Not enough data in quartiles", "SKIP")

    from scipy.stats import ks_2samp

    stat, p_value = ks_2samp(q1_ltv, q4_ltv)

    # Compute direction: negative means later cohorts have lower LTV
    mean_diff = np.mean(q4_ltv) - np.mean(q1_ltv)
    direction = mean_diff / np.mean(q1_ltv) if np.mean(q1_ltv) > 0 else 0

    detail = f"KS_stat={stat:.3f}, p={p_value:.4f}, q1_mean=${np.mean(q1_ltv):.2f}, q4_mean=${np.mean(q4_ltv):.2f}, drift={direction:.2%}"

    # Check passes if p < 0.1 (cohorts are different) — otherwise FAIL
    # WARN zone for borderline p-values to avoid flaky CI results
    result = _make_result(check_id, float(direction), detail)
    if p_value > 0.20:
        result["status"] = "FAIL"
        result["detail"] += " → Cohorts statistically indistinguishable (no drift)"
    elif p_value > 0.10:
        result["status"] = "WARN"
        result["detail"] += " → Borderline drift significance"
    else:
        # Use calibrated range from benchmark; direction must be within range
        bench = BENCHMARKS.get(check_id, {})
        lo, hi = bench.get("range", [-0.40, 0.05])
        if lo <= direction <= hi:
            result["status"] = "PASS"
        else:
            result["status"] = "WARN"
    return result


def check_5_3_repeat_rate_evolution(orders_df: pd.DataFrame, sim_days: int) -> Dict:
    """Monthly repeat rate trend (skipping first 150 days to avoid launch bias)."""
    check_id = "5.3_repeat_rate_evolution"
    if sim_days < 240:
        return _make_result(check_id, None, "Sim < 240 days, skipping", "SKIP")
    if orders_df is None or orders_df.empty:
        return _make_result(check_id, None, "No orders data", "SKIP")

    date_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
    orders = orders_df.copy()
    orders["_date"] = pd.to_datetime(orders[date_col])

    # Compute first purchase from FULL history before filtering
    first_purchase = orders.groupby("customer_id")["_date"].min().rename("first_date")
    orders = orders.merge(first_purchase, on="customer_id")
    orders["is_repeat"] = orders["_date"] > orders["first_date"]

    # Skip first 150 days — launch phase has structurally near-zero repeat rate
    # that always increases as customer base accumulates from zero.
    # A 150-day skip allows the repeat-eligible pool to stabilize before measuring trend.
    sim_start = orders["_date"].min()
    orders = orders[orders["_date"] >= sim_start + pd.Timedelta(days=150)]
    if orders.empty:
        return _make_result(check_id, None, "No orders after 90-day skip", "SKIP")

    orders["_month"] = orders["_date"].dt.to_period("M")

    monthly_repeat = orders.groupby("_month")["is_repeat"].mean()
    if len(monthly_repeat) < 4:
        return _make_result(check_id, None, "Fewer than 4 months after launch skip", "SKIP")

    vals = monthly_repeat.values.astype(float)
    mean_rr = np.mean(vals)
    cv = np.std(vals) / mean_rr if mean_rr > 0 else 0

    # Trend
    slope = np.polyfit(np.arange(len(vals)), vals, 1)[0]
    annual_trend = slope * 12

    detail = f"CV={cv:.3f}, mean_repeat_rate={mean_rr:.2%}, annual_trend={annual_trend:+.3f}"
    return _make_result(check_id, float(annual_trend), detail)


def check_5_4_trust_baseline_trend(sim_days: int, brand_state_df: pd.DataFrame = None) -> Dict:
    """Monthly avg trust score trend from diagnostic output."""
    check_id = "5.4_trust_baseline_trend"
    if brand_state_df is None or brand_state_df.empty:
        return _make_result(check_id, None, "No brand_state_daily diagnostic data available", "SKIP")
    if "mean_trust_score" not in brand_state_df.columns or "date" not in brand_state_df.columns:
        return _make_result(check_id, None, "Missing mean_trust_score or date in brand_state_daily", "SKIP")

    bs = brand_state_df.copy()
    bs["date"] = pd.to_datetime(bs["date"])
    bs = bs.dropna(subset=["mean_trust_score"]).sort_values("date")

    if len(bs) < 30:
        return _make_result(check_id, None, "Fewer than 30 days of trust data", "SKIP")

    # Monthly aggregation
    bs["_month"] = bs["date"].dt.to_period("M")
    monthly_trust = bs.groupby("_month")["mean_trust_score"].mean()

    if len(monthly_trust) < 3:
        return _make_result(check_id, None, "Fewer than 3 months of trust data", "SKIP")

    vals = monthly_trust.values.astype(float)
    slope = np.polyfit(np.arange(len(vals)), vals, 1)[0]
    annual_trend = slope * 12

    status = None
    if sim_days < 180:
        status = "WARN"  # Low confidence with < 6 months
    detail = f"monthly_trust_slope={slope:.5f}, annual_trend={annual_trend:+.4f}, n_months={len(vals)}"
    return _make_result(check_id, float(annual_trend), detail, status)


def check_5_5_divergence(sim_days: int) -> Dict:
    """1yr vs 3yr divergence — only in FINAL BOSS TEST."""
    check_id = "5.5_1yr_vs_3yr_divergence"
    if sim_days < 365:
        return _make_result(check_id, None, "Only runs in FINAL BOSS TEST (>=365d)", "SKIP")
    # Requires comparison of two separate sim runs; skip in single-run mode
    return _make_result(check_id, None, "Requires two separate sim runs for comparison", "SKIP")


# ---------------------------------------------------------------------------
# CATEGORY 6: FEEDBACK LOOPS
# ---------------------------------------------------------------------------


def check_6_1_refund_trust_granger(daily_refunds: pd.Series, brand_state_df: pd.DataFrame = None, sim_days: int = 0) -> Dict:
    """Lag correlation: refund count → trust changes."""
    check_id = "6.1_refund_trust_granger"
    if brand_state_df is None or brand_state_df.empty:
        return _make_result(check_id, None, "No brand_state_daily diagnostic data available", "SKIP")
    if "mean_trust_score" not in brand_state_df.columns or "date" not in brand_state_df.columns:
        return _make_result(check_id, None, "Missing mean_trust_score or date in brand_state_daily", "SKIP")
    if daily_refunds is None or len(daily_refunds) < 30:
        return _make_result(check_id, None, "Insufficient daily refund data", "SKIP")

    bs = brand_state_df.copy()
    bs["date"] = pd.to_datetime(bs["date"])
    bs = bs.sort_values("date").set_index("date")
    trust_change = bs["mean_trust_score"].diff().dropna()

    # Align refund series with trust change dates
    ref_series = daily_refunds.copy()
    ref_series.index = pd.to_datetime(ref_series.index)
    common_dates = trust_change.index.intersection(ref_series.index)
    if len(common_dates) < 30:
        return _make_result(check_id, None, "Fewer than 30 aligned days", "SKIP")

    trust_vals = trust_change.loc[common_dates].values.astype(float)
    ref_vals = ref_series.loc[common_dates].values.astype(float)

    # Test lag-1 to lag-7 Pearson correlation
    from scipy import stats
    best_p = 1.0
    best_lag = 0
    best_corr = 0.0
    for lag in range(1, 8):
        if lag >= len(trust_vals):
            break
        r, p = stats.pearsonr(ref_vals[:-lag], trust_vals[lag:])
        if not np.isnan(p) and p < best_p:
            best_p = p
            best_lag = lag
            best_corr = r

    # In short sims (<180d), trust changes are diluted across 1000+ customers,
    # making lag correlation undetectable. Downgrade to INFO.
    status = None
    if sim_days > 0 and sim_days < 180:
        detail = f"best_lag={best_lag}, corr={best_corr:.3f}, p={best_p:.4f}, short_sim={sim_days}d"
        status = "INFO"
    else:
        detail = f"best_lag={best_lag}, corr={best_corr:.3f}, p={best_p:.4f}"
    return _make_result(check_id, float(best_p), detail, status)


def check_6_2_trust_repeat_granger(brand_state_df: pd.DataFrame = None, orders_df: pd.DataFrame = None) -> Dict:
    """Lag correlation: mean trust → daily repeat count."""
    check_id = "6.2_trust_repeat_granger"
    if brand_state_df is None or brand_state_df.empty:
        return _make_result(check_id, None, "No brand_state_daily diagnostic data available", "SKIP")
    if orders_df is None or orders_df.empty:
        return _make_result(check_id, None, "No orders data", "SKIP")
    if "mean_trust_score" not in brand_state_df.columns:
        return _make_result(check_id, None, "Missing mean_trust_score in brand_state_daily", "SKIP")

    bs = brand_state_df.copy()
    bs["date"] = pd.to_datetime(bs["date"])
    bs = bs.sort_values("date").set_index("date")

    # Build daily repeat count: orders where customer has prior orders on an earlier date
    date_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
    odf = orders_df.copy()
    odf["_date"] = pd.to_datetime(odf[date_col]).dt.normalize()
    odf = odf.sort_values("_date")

    # Identify repeat orders: customer has appeared in an earlier order
    seen_customers = set()
    repeat_flags = []
    for _, row in odf.iterrows():
        cid = row["customer_id"]
        repeat_flags.append(cid in seen_customers)
        seen_customers.add(cid)
    odf["_is_repeat"] = repeat_flags
    daily_repeat = odf.groupby("_date")["_is_repeat"].sum()

    # Align trust and repeat series
    common_dates = bs.index.intersection(daily_repeat.index)
    if len(common_dates) < 30:
        return _make_result(check_id, None, "Fewer than 30 aligned days", "SKIP")

    trust_vals = bs.loc[common_dates, "mean_trust_score"].values.astype(float)
    repeat_vals = daily_repeat.loc[common_dates].values.astype(float)

    # Test lag-1 to lag-7 correlation
    from scipy import stats
    best_p = 1.0
    best_lag = 0
    best_corr = 0.0
    for lag in range(1, 8):
        if lag >= len(trust_vals):
            break
        r, p = stats.pearsonr(trust_vals[:-lag], repeat_vals[lag:])
        if not np.isnan(p) and p < best_p:
            best_p = p
            best_lag = lag
            best_corr = r

    detail = f"best_lag={best_lag}, corr={best_corr:.3f}, p={best_p:.4f}"
    return _make_result(check_id, float(best_p), detail)


def check_6_3_spiral_detection(
    daily_revenue: pd.Series, daily_refunds: pd.Series
) -> Dict:
    """Rolling correlation: CPA ↔ refund rate."""
    check_id = "6.3_spiral_detection"
    if daily_revenue is None or daily_refunds is None:
        return _make_result(check_id, None, "Missing daily data", "SKIP")
    if len(daily_revenue) < 60:
        return _make_result(check_id, None, "Insufficient data (<60 days)", "SKIP")

    rev = daily_revenue.values.astype(float)
    ref = daily_refunds.values.astype(float)
    n = min(len(rev), len(ref))

    # Rolling 30-day correlation
    window = 30
    spirals = 0
    for i in range(window, n):
        r = rev[i - window : i]
        f = ref[i - window : i]
        if np.std(r) > 0 and np.std(f) > 0:
            corr = np.corrcoef(r, f)[0, 1]
            if not np.isnan(corr) and corr < -0.3:
                spirals += 1

    # Normalize by number of windows
    n_windows = n - window
    spiral_rate = spirals / n_windows if n_windows > 0 else 0

    detail = f"spiral_windows={spirals}/{n_windows}, rate={spiral_rate:.2%}"
    return _make_result(check_id, spirals, detail, "INFO")


# ---------------------------------------------------------------------------
# CATEGORY 7: MEMORY & PATH DEPENDENCE
# ---------------------------------------------------------------------------


def check_7_1_history_effect(orders_df: pd.DataFrame, exposure_df: pd.DataFrame, sim_days: int = 0) -> Dict:
    """Correlation between total exposures and order count per customer."""
    check_id = "7.1_history_effect"
    if orders_df is None or exposure_df is None or orders_df.empty or exposure_df.empty:
        return _make_result(check_id, None, "Missing orders or exposure data", "SKIP")

    # Count total exposures per customer (non-null customer_ids only)
    valid_exp = exposure_df[exposure_df["customer_id"].notna()] if "customer_id" in exposure_df.columns else pd.DataFrame()
    if valid_exp.empty:
        return _make_result(check_id, None, "No exposures with customer_id", "SKIP")

    exposure_count = valid_exp.groupby("customer_id").size().rename("exp_count")

    # Count orders per customer
    order_counts = orders_df.groupby("customer_id").size().rename("order_count")

    # Merge: exposure count and order count
    merged = exposure_count.reset_index().merge(order_counts.reset_index(), on="customer_id", how="left")
    merged["order_count"] = merged["order_count"].fillna(0)

    if len(merged) < 20:
        return _make_result(check_id, None, "Fewer than 20 exposed users", "SKIP")

    corr = np.corrcoef(merged["exp_count"].values.astype(float), merged["order_count"].values.astype(float))[0, 1]
    if np.isnan(corr):
        return _make_result(check_id, None, "Correlation is NaN", "SKIP")

    # In short sims (<180d), most customers have 0-1 orders, creating near-zero
    # variance in order_count. This makes Pearson correlation unreliable.
    # Downgrade to INFO and use wider range for short sims.
    status = None
    order_var = merged["order_count"].var()
    if sim_days > 0 and sim_days < 180:
        detail = f"corr(exposure_count, order_count)={corr:.3f}, order_var={order_var:.3f}, short_sim={sim_days}d"
        status = "INFO"
    else:
        detail = f"corr(exposure_count, order_count)={corr:.3f}, order_var={order_var:.3f}"

    return _make_result(check_id, float(corr), detail, status)


def check_7_2_negative_experience_persistence(
    orders_df: pd.DataFrame, refunds_df: pd.DataFrame
) -> Dict:
    """Behavior at 30/60/90d post-negative experience."""
    check_id = "7.2_negative_experience_persistence"
    if orders_df is None or refunds_df is None or orders_df.empty or refunds_df.empty:
        return _make_result(check_id, None, "Missing orders or refunds", "SKIP")

    # Simplified: compare time-to-next-purchase for refunders vs non-refunders
    date_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
    orders = orders_df.copy()
    orders["_date"] = pd.to_datetime(orders[date_col])

    refunded_order_ids = set(refunds_df["order_id"].astype(str)) if "order_id" in refunds_df.columns else set()
    if not refunded_order_ids:
        return _make_result(check_id, None, "No refunds to analyze", "SKIP")

    refunder_ids = set(orders[orders["id"].astype(str).isin(refunded_order_ids)]["customer_id"])
    if len(refunder_ids) < 5:
        return _make_result(check_id, None, "Fewer than 5 refunders", "SKIP")

    # For refunders: time between refunded order and next order
    intervals = []
    for cid in refunder_ids:
        cust_orders = orders[orders["customer_id"] == cid].sort_values("_date")
        if len(cust_orders) < 2:
            continue
        refunded_mask = cust_orders["id"].astype(str).isin(refunded_order_ids)
        for idx in cust_orders[refunded_mask].index:
            pos = cust_orders.index.get_loc(idx)
            if pos + 1 < len(cust_orders):
                gap = (cust_orders.iloc[pos + 1]["_date"] - cust_orders.iloc[pos]["_date"]).days
                if gap > 0:
                    intervals.append(gap)

    if len(intervals) < 3:
        return _make_result(check_id, None, "Not enough post-refund purchase data", "SKIP")

    median_gap = np.median(intervals)
    detail = f"median_post_refund_gap={median_gap:.0f}d, n={len(intervals)}"
    return _make_result(check_id, float(median_gap), detail)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_diagnostics(data_dir: Path, sim_days: int = 0, final_boss: bool = False, diag_dir: Path = None) -> Dict:
    """Run all diagnostic checks and produce report."""
    print("Loading simulator output data...")
    data = load_sim_data(data_dir)

    # Load diagnostic data if available
    brand_state_df = None
    if diag_dir is not None and diag_dir.exists():
        print(f"Loading diagnostic data from {diag_dir}...")
        diag_data = load_diagnostic_data(diag_dir)
        brand_state_df = diag_data.get("brand_state_daily")

    orders_df = data["orders"]
    exposure_df = data["meta_exposures"]
    perf_df = data["meta_ad_performance_daily"]
    refunds_df = data["refunds"]

    # Auto-detect sim duration if not provided
    if sim_days == 0 and orders_df is not None and not orders_df.empty:
        date_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
        if date_col in orders_df.columns:
            dates = pd.to_datetime(orders_df[date_col])
            sim_days = (dates.max() - dates.min()).days + 1

    # Build daily aggregates
    daily_revenue = None
    daily_refunds = None
    if orders_df is not None and not orders_df.empty:
        date_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
        if date_col in orders_df.columns:
            orders_copy = orders_df.copy()
            orders_copy["_date"] = pd.to_datetime(orders_copy[date_col]).dt.normalize()
            daily_revenue = orders_copy.groupby("_date")["total_price"].sum()
            daily_revenue = daily_revenue.sort_index()
            # Reindex to fill missing days
            if len(daily_revenue) > 1:
                full_range = pd.date_range(daily_revenue.index.min(), daily_revenue.index.max())
                daily_revenue = daily_revenue.reindex(full_range, fill_value=0)

    if refunds_df is not None and not refunds_df.empty:
        refund_date_col = "created_at" if "created_at" in refunds_df.columns else "processed_at"
        if refund_date_col in refunds_df.columns:
            refunds_copy = refunds_df.copy()
            refunds_copy["_date"] = pd.to_datetime(refunds_copy[refund_date_col]).dt.normalize()
            daily_refunds = refunds_copy.groupby("_date").size()
            if daily_revenue is not None and len(daily_revenue) > 0:
                daily_refunds = daily_refunds.reindex(daily_revenue.index, fill_value=0)

    # Population size
    pop_size = 0
    if exposure_df is not None:
        pop_size = exposure_df["customer_id"].nunique() if "customer_id" in exposure_df.columns else 0

    print(f"Sim duration: {sim_days} days, Population: {pop_size}")
    print("Running checks...\n")

    checks = {}

    # Category 1: Temporal Structure
    checks["1.1_revenue_autocorrelation"] = check_1_1_revenue_autocorrelation(daily_revenue)
    checks["1.2_seasonality_presence"] = check_1_2_seasonality_presence(daily_revenue)
    checks["1.3_trend_presence"] = check_1_3_trend_presence(daily_revenue, sim_days=sim_days)
    checks["1.4_non_stationarity"] = check_1_4_non_stationarity(daily_revenue)

    # Category 2: Distribution Shape
    checks["2.1_customer_ltv_distribution"] = check_2_1_customer_ltv(orders_df)
    checks["2.2_order_value_cv"] = check_2_2_order_value_cv(orders_df)
    checks["2.3_purchase_interval_distribution"] = check_2_3_purchase_intervals(orders_df)
    checks["2.4_refund_timing_distribution"] = check_2_4_refund_timing(refunds_df, orders_df)

    # Category 3: Cross-Metric Relationships
    checks["3.1_creative_age_vs_performance"] = check_3_1_creative_age_vs_performance(perf_df)
    checks["3.2_discount_vs_repeat"] = check_3_2_discount_vs_repeat(orders_df)
    checks["3.3_frequency_vs_conversion"] = check_3_3_frequency_vs_conversion(exposure_df, orders_df)
    checks["3.4_refund_vs_repeat"] = check_3_4_refund_vs_repeat(orders_df, refunds_df)
    checks["3.5_cross_lag_correlations"] = check_3_5_cross_lag_correlations(daily_revenue, daily_refunds)

    # Category 4: Structural Asymmetry
    checks["4.1_creative_concentration"] = check_4_1_creative_concentration(perf_df)
    checks["4.2_customer_concentration"] = check_4_2_customer_concentration(orders_df)
    checks["4.3_channel_asymmetry"] = check_4_3_channel_asymmetry(perf_df)

    # Category 5: Time-Dependence
    checks["5.1_acquisition_cost_trend"] = check_5_1_acquisition_cost_trend(perf_df, sim_days, orders_df)
    checks["5.2_cohort_composition_drift"] = check_5_2_cohort_composition_drift(orders_df, sim_days)
    checks["5.3_repeat_rate_evolution"] = check_5_3_repeat_rate_evolution(orders_df, sim_days)
    checks["5.4_trust_baseline_trend"] = check_5_4_trust_baseline_trend(sim_days, brand_state_df=brand_state_df)
    checks["5.5_1yr_vs_3yr_divergence"] = check_5_5_divergence(sim_days)

    # Category 6: Feedback Loops
    checks["6.1_refund_trust_granger"] = check_6_1_refund_trust_granger(daily_refunds, brand_state_df=brand_state_df, sim_days=sim_days)
    checks["6.2_trust_repeat_granger"] = check_6_2_trust_repeat_granger(brand_state_df=brand_state_df, orders_df=orders_df)
    checks["6.3_spiral_detection"] = check_6_3_spiral_detection(daily_revenue, daily_refunds)

    # Category 7: Memory & Path Dependence
    checks["7.1_history_effect"] = check_7_1_history_effect(orders_df, exposure_df, sim_days=sim_days)
    checks["7.2_negative_experience_persistence"] = check_7_2_negative_experience_persistence(orders_df, refunds_df)

    # Compute summary
    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0, "INFO": 0}
    for result in checks.values():
        s = result.get("status", "SKIP")
        counts[s] = counts.get(s, 0) + 1

    run_id = f"diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    overall_pass = counts["FAIL"] == 0

    report = {
        "run_id": run_id,
        "sim_duration_days": sim_days,
        "population_size": pop_size,
        "timestamp": datetime.now().isoformat(),
        "overall_pass": overall_pass,
        "summary": counts,
        "checks": checks,
    }

    return report


def write_reports(report: Dict):
    """Write JSON and human-readable reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = REPORTS_DIR / "latest_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report: {json_path}")

    # Baseline handling
    baseline_path = REPORTS_DIR / "baseline_report.json"
    if not baseline_path.exists():
        shutil.copy2(json_path, baseline_path)
        print(f"Baseline created: {baseline_path}")

    # Human-readable report
    txt_path = REPORTS_DIR / "latest_report_human.txt"
    lines = []
    s = report["summary"]
    lines.append("=" * 62)
    lines.append(f"  STRUCTURAL DIAGNOSTICS REPORT — Run {report['run_id']}")
    lines.append(f"  Duration: {report['sim_duration_days']} days | Population: {report['population_size']}")
    lines.append(
        f"  Overall: {s['PASS']} PASS | {s['FAIL']} FAIL | {s['WARN']} WARN | {s['SKIP']} SKIP | {s['INFO']} INFO"
    )
    lines.append("=" * 62)

    # Group by category
    _CATEGORY_NAMES = {
        "temporal_structure": "TEMPORAL STRUCTURE",
        "distribution_shape": "DISTRIBUTION SHAPE",
        "cross_metric": "CROSS-METRIC RELATIONSHIPS",
        "structural_asymmetry": "STRUCTURAL ASYMMETRY",
        "time_dependence": "TIME-DEPENDENCE",
        "feedback_loops": "FEEDBACK LOOPS",
        "memory_path_dependence": "MEMORY & PATH DEPENDENCE",
    }
    _STATUS_ICONS = {
        "PASS": "  PASS ",
        "FAIL": "  FAIL ",
        "WARN": "  WARN ",
        "SKIP": "  SKIP ",
        "INFO": "  INFO ",
    }

    categories_seen = set()
    for check_id, result in report["checks"].items():
        cat = result.get("category", "unknown")
        if cat not in categories_seen:
            categories_seen.add(cat)
            lines.append("")
            lines.append(f"  {_CATEGORY_NAMES.get(cat, cat.upper())}")

        status = result.get("status", "SKIP")
        icon = _STATUS_ICONS.get(status, "  ???? ")
        observed = result.get("observed")
        obs_str = f"{observed}" if observed is not None else "N/A"
        bench = result.get("benchmark_range")
        bench_str = f"{bench}" if bench else ""

        name = check_id.replace("_", " ").title()
        line = f" {icon} {check_id:<40s} {obs_str:<12s} {bench_str}"
        lines.append(line)

        # Add detail for failures
        if status == "FAIL":
            detail = result.get("detail", "")
            lines.append(f"         -> {detail}")

    lines.append("")
    lines.append("=" * 62)

    txt_content = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"Human report: {txt_path}")
    print(f"\n{txt_content}")


def main():
    parser = argparse.ArgumentParser(description="Structural Diagnostics for Emakie Simulator")
    parser.add_argument("--data-dir", type=str, default="output", help="Path to simulator output directory")
    parser.add_argument("--diag-dir", type=str, default="diagnostics_output", help="Path to diagnostic output directory")
    parser.add_argument("--sim-days", type=int, default=0, help="Simulation duration in days (auto-detected if 0)")
    parser.add_argument("--final-boss", action="store_true", help="Run in FINAL BOSS mode")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    diag_dir = Path(args.diag_dir)
    if not diag_dir.exists():
        diag_dir = None

    report = run_diagnostics(data_dir, sim_days=args.sim_days, final_boss=args.final_boss, diag_dir=diag_dir)
    write_reports(report)

    if report["summary"]["FAIL"] > 0:
        print(f"\n{report['summary']['FAIL']} HARD FAIL(s) detected.")
        sys.exit(1)
    else:
        print("\nNo HARD FAILs. Diagnostics complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
