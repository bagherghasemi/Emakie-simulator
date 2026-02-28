"""
Publish static dimension tables from simulator DataFrames into BigQuery.

Writes once per simulation using WRITE_TRUNCATE. Used by main.py after
generating static entities and before the daily simulation loop.
Conforms to schema contract (see loaders/schema_contract.py) when available.
"""

from typing import TYPE_CHECKING

import pandas as pd
from google.cloud import bigquery

from emakie_simulator.loaders import schema_contract

if TYPE_CHECKING:
    from google.cloud.bigquery import Client


def _write_table(client: "Client", df: pd.DataFrame, table_id: str, table_name: str) -> int:
    """Write DataFrame to BigQuery table using WRITE_TRUNCATE. Returns row count."""
    schema_contract.validate_dataframe_against_contract(df, table_name)
    df = df.copy()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    for col in datetime_cols:
        df[col] = df[col].dt.floor("us")
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    if df.empty:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        return 0
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    return len(df)


def load_static_entities(
    client: "Client",
    project_id: str,
    dataset_meta: str,
    dataset_shopify: str,
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    creatives_df: pd.DataFrame,
    ad_accounts_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    adsets_df: pd.DataFrame,
    ads_df: pd.DataFrame,
) -> None:
    """Load static dimension tables into BigQuery. Uses WRITE_TRUNCATE per scenario."""
    mappings = [
        (ad_accounts_df, dataset_meta, "meta_ad_accounts"),
        (campaigns_df, dataset_meta, "meta_campaigns"),
        (adsets_df, dataset_meta, "meta_ad_sets"),
        (ads_df, dataset_meta, "meta_ads"),
        (creatives_df, dataset_meta, "meta_creatives"),
        (customers_df, dataset_shopify, "shopify_customers"),
        (products_df, dataset_shopify, "shopify_products"),
        (variants_df, dataset_shopify, "shopify_product_variants"),
    ]
    for df, dataset_id, table_name in mappings:
        table_id = f"{project_id}.{dataset_id}.{table_name}"
        n = _write_table(client, df, table_id, table_name)
        print(f"Loaded {n} rows into {dataset_id}.{table_name}")
