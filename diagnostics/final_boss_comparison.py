#!/usr/bin/env python3
"""
Final Boss Test: Year 1 vs Year 3 structural comparison.

Loads 3 years of simulated parquet data (orders, ad performance, refunds),
splits into Year 1 (2024) and Year 3 (2026), computes structural metrics for
each period, and checks that at least 3 metrics diverge by more than 10%.

A healthy simulator should produce *evolving* data -- if Year 1 and Year 3
look identical across every metric, the simulation is too static.

Usage:
    python diagnostics/final_boss_comparison.py [--data-dir output/]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_parquets(directory: str, year: int) -> pd.DataFrame:
    """Load all parquet files for a given year from *directory*.

    Files are expected to be named ``YYYY-MM-DD.parquet``.
    Returns an empty DataFrame when no matching files are found.
    """
    pattern = os.path.join(directory, f"{year}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def load_year_data(data_dir: str, year: int):
    """Return (orders, ads, refunds) DataFrames for *year*."""
    orders = _load_parquets(os.path.join(data_dir, "orders"), year)
    ads = _load_parquets(os.path.join(data_dir, "meta_ad_performance_daily"), year)
    refunds = _load_parquets(os.path.join(data_dir, "refunds"), year)
    return orders, ads, refunds


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _autocorrelation_lag1(series: pd.Series) -> float:
    """Pearson autocorrelation at lag-1 for a daily time-series."""
    if len(series) < 3:
        return float("nan")
    return float(series.autocorr(lag=1))


def _gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient for an array of non-negative values."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def compute_metrics(
    orders: pd.DataFrame,
    ads: pd.DataFrame,
    refunds: pd.DataFrame,
) -> dict[str, float]:
    """Compute all structural metrics for one year-slice.

    Returns a dict mapping metric name -> value.
    """
    metrics: dict[str, float] = {}

    # -- Daily revenue -------------------------------------------------------
    if not orders.empty and "created_at" in orders.columns:
        orders = orders.copy()
        orders["_date"] = pd.to_datetime(orders["created_at"]).dt.date
        daily_rev = orders.groupby("_date")["total_price"].sum().sort_index()
        metrics["daily_revenue_mean"] = float(daily_rev.mean())
        metrics["daily_revenue_std"] = float(daily_rev.std())
        metrics["daily_revenue_autocorr_lag1"] = _autocorrelation_lag1(daily_rev)
    else:
        metrics["daily_revenue_mean"] = float("nan")
        metrics["daily_revenue_std"] = float("nan")
        metrics["daily_revenue_autocorr_lag1"] = float("nan")

    # -- Orders per day ------------------------------------------------------
    if not orders.empty:
        daily_orders = orders.groupby("_date").size().sort_index()
        metrics["orders_per_day_mean"] = float(daily_orders.mean())
    else:
        metrics["orders_per_day_mean"] = float("nan")

    # -- Conversion rate (orders / clicks) -----------------------------------
    if not ads.empty and "clicks" in ads.columns:
        total_clicks = int(ads["clicks"].sum())
    else:
        total_clicks = 0
    total_orders = len(orders) if not orders.empty else 0
    if total_clicks > 0:
        metrics["conversion_rate"] = total_orders / total_clicks
    else:
        metrics["conversion_rate"] = float("nan")

    # -- Refund rate (refunds / orders) --------------------------------------
    total_refunds = len(refunds) if not refunds.empty else 0
    if total_orders > 0:
        metrics["refund_rate"] = total_refunds / total_orders
    else:
        metrics["refund_rate"] = float("nan")

    # -- CPA: monthly spend / new customers ----------------------------------
    if not ads.empty and not orders.empty:
        ads = ads.copy()
        ads["_month"] = pd.to_datetime(ads["date"]).dt.to_period("M")
        monthly_spend = ads.groupby("_month")["spend"].sum()

        orders_sorted = orders.sort_values("created_at")
        first_order = orders_sorted.drop_duplicates(subset="customer_id", keep="first").copy()
        first_order["_month"] = pd.to_datetime(first_order["created_at"]).dt.to_period("M")
        monthly_new_custs = first_order.groupby("_month").size()

        common_months = monthly_spend.index.intersection(monthly_new_custs.index)
        if len(common_months) > 0:
            cpa_series = monthly_spend.loc[common_months] / monthly_new_custs.loc[common_months].replace(0, np.nan)
            metrics["cpa_mean_monthly"] = float(cpa_series.mean())
        else:
            metrics["cpa_mean_monthly"] = float("nan")
    else:
        metrics["cpa_mean_monthly"] = float("nan")

    # -- LTV Gini coefficient ------------------------------------------------
    if not orders.empty:
        ltv = orders.groupby("customer_id")["total_price"].sum().values
        metrics["ltv_gini"] = _gini(ltv)
    else:
        metrics["ltv_gini"] = float("nan")

    # -- Repeat rate (fraction of orders from repeat customers) ---------------
    if not orders.empty:
        cust_order_counts = orders.groupby("customer_id").size()
        repeat_customers = set(cust_order_counts[cust_order_counts > 1].index)
        orders_from_repeat = orders["customer_id"].isin(repeat_customers).sum()
        metrics["repeat_rate"] = float(orders_from_repeat) / total_orders if total_orders > 0 else float("nan")
    else:
        metrics["repeat_rate"] = float("nan")

    # -- Customer concentration (top 10% revenue share) ----------------------
    if not orders.empty:
        cust_rev = orders.groupby("customer_id")["total_price"].sum().sort_values(ascending=False)
        n_top = max(1, int(np.ceil(len(cust_rev) * 0.10)))
        top_rev = cust_rev.iloc[:n_top].sum()
        metrics["customer_concentration_top10pct"] = float(top_rev / cust_rev.sum()) if cust_rev.sum() > 0 else float("nan")
    else:
        metrics["customer_concentration_top10pct"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Comparison & scoring
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "daily_revenue_mean": "Daily Revenue Mean",
    "daily_revenue_std": "Daily Revenue Std Dev",
    "daily_revenue_autocorr_lag1": "Daily Revenue Autocorr(1)",
    "orders_per_day_mean": "Orders/Day Mean",
    "conversion_rate": "Conversion Rate (orders/clicks)",
    "refund_rate": "Refund Rate",
    "cpa_mean_monthly": "CPA (monthly mean)",
    "ltv_gini": "LTV Gini Coefficient",
    "repeat_rate": "Repeat Rate",
    "customer_concentration_top10pct": "Customer Concentration (top 10%)",
}


def pct_diff(a: float, b: float) -> float:
    """Symmetric percentage difference: |a-b| / ((|a|+|b|)/2) * 100.

    Returns NaN when both values are zero or either is NaN.
    """
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    denom = (abs(a) + abs(b)) / 2.0
    if denom == 0:
        return 0.0
    return abs(a - b) / denom * 100.0


def print_comparison(y1: dict[str, float], y3: dict[str, float]) -> int:
    """Print a formatted table comparing Year 1 and Year 3 metrics.

    Returns the divergence score (number of metrics diverging by >10%).
    """
    divergence_score = 0
    diverged_metrics: list[str] = []

    header = f"{'Metric':<35} {'Year 1':>14} {'Year 3':>14} {'% Diff':>10} {'Diverged':>10}"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  FINAL BOSS TEST: Year 1 (2024) vs Year 3 (2026)")
    print("=" * len(header))
    print(header)
    print(sep)

    for key in METRIC_LABELS:
        label = METRIC_LABELS[key]
        v1 = y1.get(key, float("nan"))
        v3 = y3.get(key, float("nan"))
        diff = pct_diff(v1, v3)

        if np.isnan(diff):
            flag = "N/A"
        elif diff > 10.0:
            flag = "YES"
            divergence_score += 1
            diverged_metrics.append(label)
        else:
            flag = "no"

        def _fmt(v: float) -> str:
            if np.isnan(v):
                return "N/A"
            if abs(v) >= 1000:
                return f"{v:,.2f}"
            return f"{v:.6f}"

        diff_str = f"{diff:.1f}%" if not np.isnan(diff) else "N/A"
        print(f"{label:<35} {_fmt(v1):>14} {_fmt(v3):>14} {diff_str:>10} {flag:>10}")

    print(sep)
    print()
    print(f"Divergence score: {divergence_score} / {len(METRIC_LABELS)}")
    if diverged_metrics:
        print("Diverged metrics:")
        for m in diverged_metrics:
            print(f"  - {m}")
    print()

    return divergence_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final Boss Test: Year 1 vs Year 3 structural comparison."
    )
    parser.add_argument(
        "--data-dir",
        default="output/",
        help="Root directory containing orders/, meta_ad_performance_daily/, refunds/ subdirs (default: output/)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    # Validate that expected subdirectories exist
    for sub in ("orders", "meta_ad_performance_daily", "refunds"):
        subpath = os.path.join(data_dir, sub)
        if not os.path.isdir(subpath):
            print(f"ERROR: Expected subdirectory not found: {subpath}", file=sys.stderr)
            sys.exit(1)

    print("Loading Year 1 (2024) data ...")
    orders_y1, ads_y1, refunds_y1 = load_year_data(data_dir, 2024)
    print(
        f"  orders={len(orders_y1):,}  ads={len(ads_y1):,}  refunds={len(refunds_y1):,}"
    )

    print("Loading Year 3 (2026) data ...")
    orders_y3, ads_y3, refunds_y3 = load_year_data(data_dir, 2026)
    print(
        f"  orders={len(orders_y3):,}  ads={len(ads_y3):,}  refunds={len(refunds_y3):,}"
    )

    print("\nComputing metrics ...")
    metrics_y1 = compute_metrics(orders_y1, ads_y1, refunds_y1)
    metrics_y3 = compute_metrics(orders_y3, ads_y3, refunds_y3)

    divergence_score = print_comparison(metrics_y1, metrics_y3)

    threshold = 3
    if divergence_score >= threshold:
        print(f"PASS  -- divergence_score ({divergence_score}) >= {threshold}")
        print("The simulator produces structurally evolving data over 3 years.")
    else:
        print(f"FAIL  -- divergence_score ({divergence_score}) < {threshold}")
        print("The simulator data is too static between Year 1 and Year 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()
