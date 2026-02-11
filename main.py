"""
Multi-day behavioral simulation.

Setup (once): load config, generate static entities (customers, products,
variants, creatives, campaigns, adsets, ads). Then iterate day by day from
config.simulation.start_date to end_date: simulate_daily_exposure → extract
clicks → simulate_purchases_from_clicks → simulate_refunds. Write daily
parquet files to output/ subfolders; no in-memory accumulation. Local only;
no BigQuery.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from generators.aftermath import simulate_refunds
from generators.commerce import (
    generate_products,
    generate_variants,
    simulate_purchases_from_clicks,
)
from generators.humans import generate_customers
from generators.meta import (
    generate_ads,
    generate_adsets,
    generate_campaigns,
    generate_creatives,
    simulate_daily_exposure,
)

OUTPUT_ROOT = Path(__file__).parent / "output"
SUBFOLDERS = (
    "meta_exposures",
    "orders",
    "line_items",
    "transactions",
    "refunds",
)


def _ensure_output_dirs() -> None:
    """Create output/ and subfolders if they do not exist (append-safe)."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in SUBFOLDERS:
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def _write_parquet(df, folder: str, date_str: str) -> None:
    path = OUTPUT_ROOT / folder / f"{date_str}.parquet"
    df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-day behavioral simulation.")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to config YAML (default: config.yaml in script directory)",
    )
    args = parser.parse_args()
    config_path = (
        Path(args.config).resolve()
        if args.config
        else Path(__file__).parent / "config.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    sim = config.get("simulation") or {}
    start_str = sim.get("start_date", "2023-01-01")
    end_str = sim.get("end_date", "2024-01-01")
    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    if start_date > end_date:
        raise ValueError(f"start_date {start_str} must be <= end_date {end_str}")

    _ensure_output_dirs()

    # --- Setup (once): static entities ---
    customers_df = generate_customers(config)
    products_df = generate_products(config)
    variants_df = generate_variants(products_df)
    creatives_df = generate_creatives(config)
    campaigns_df = generate_campaigns(creatives_df)
    adsets_df = generate_adsets(campaigns_df)
    ads_df = generate_ads(adsets_df, creatives_df)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")

        exposure_df = simulate_daily_exposure(
            customers_df, ads_df, creatives_df, date_str, config
        )
        n_exposures = len(exposure_df)

        clicks = exposure_df[exposure_df["clicked"] == 1]
        n_clicks = len(clicks)

        orders_df, line_items_df, transactions_df = simulate_purchases_from_clicks(
            clicks, customers_df, variants_df, config
        )
        n_orders = len(orders_df)

        refunds_df = simulate_refunds(
            orders_df,
            line_items_df,
            customers_df,
            products_df,
            variants_df,
            config,
        )
        n_refunds = len(refunds_df)

        _write_parquet(exposure_df, "meta_exposures", date_str)
        _write_parquet(orders_df, "orders", date_str)
        _write_parquet(line_items_df, "line_items", date_str)
        _write_parquet(transactions_df, "transactions", date_str)
        _write_parquet(refunds_df, "refunds", date_str)

        print(
            f"{date_str}\t exposures={n_exposures}\t clicks={n_clicks}\t "
            f"orders={n_orders}\t refunds={n_refunds}"
        )

        current += timedelta(days=1)


if __name__ == "__main__":
    main()