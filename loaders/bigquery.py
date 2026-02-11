"""
Load parquet outputs from output/ into BigQuery.

Reads project_id, dataset_meta, dataset_shopify from config.yaml.
Append mode; creates tables if missing. Logs row counts per table.
Idempotent: skips parquet files whose date is already loaded (by table max date).
"""

from datetime import date
from pathlib import Path
import time

import pandas as pd
import yaml
from google.cloud import bigquery
from google.api_core import exceptions

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"

# Meta dataset tables (from meta_exposures: impressions + clicks; others from folders if present)
META_TABLES = ("impressions", "clicks", "ads", "adsets", "campaigns", "creatives")
# Shopify dataset: one folder per table
SHOPIFY_TABLES = ("orders", "line_items", "transactions", "refunds")

# Folder -> (dataset_key, table_name); meta_exposures handled separately
FOLDER_TO_TABLE = {
    "orders": ("shopify", "orders"),
    "line_items": ("shopify", "line_items"),
    "transactions": ("shopify", "transactions"),
    "refunds": ("shopify", "refunds"),
    "ads": ("meta", "ads"),
    "adsets": ("meta", "adsets"),
    "campaigns": ("meta", "campaigns"),
    "creatives": ("meta", "creatives"),
}

# For idempotency: BigQuery expression for max date (None = load all, no skip)
FOLDER_MAX_DATE_EXPR = {
    "orders": "DATE(order_timestamp)",
    "line_items": None,
    "transactions": None,
    "refunds": "DATE(refund_timestamp)",
    "ads": None,
    "adsets": None,
    "campaigns": None,
    "creatives": None,
}


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    project_id = config.get("project_id")
    dataset_meta = config.get("dataset_meta")
    dataset_shopify = config.get("dataset_shopify")
    if not project_id or not dataset_meta or not dataset_shopify:
        raise ValueError(
            "config.yaml must define project_id, dataset_meta, and dataset_shopify"
        )
    return {
        "project_id": project_id,
        "dataset_meta": dataset_meta,
        "dataset_shopify": dataset_shopify,
    }


def _table_ref(project_id: str, dataset_id: str, table_id: str) -> str:
    return f"{project_id}.{dataset_id}.{table_id}"


def _table_exists(client: bigquery.Client, table_id: str) -> bool:
    try:
        client.get_table(table_id)
        return True
    except exceptions.NotFound:
        return False


def _get_max_date(
    client: bigquery.Client, table_id: str, date_expr: str
) -> date | None:
    """Return MAX(date) from table; None if table missing or empty."""
    if not _table_exists(client, table_id):
        return None
    query = f"SELECT MAX({date_expr}) AS max_d FROM `{table_id}`"
    job = client.query(query)
    row = next(job.result(), None)
    if row is None or row.max_d is None:
        return None
    val = row.max_d
    return val.date() if hasattr(val, "date") else val


def _load_df_append(
    client: bigquery.Client,
    df: pd.DataFrame,
    table_id: str,
    create_if_missing: bool,
) -> int:
    if df.empty:
        return 0
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=True,
    )
    if create_if_missing and not _table_exists(client, table_id):
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    backoff_seconds = [1, 2, 4, 8, 16]
    last_error = None
    for attempt in range(5):
        try:
            job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
            job.result()
            return len(df)
        except exceptions.ServiceUnavailable as e:
            last_error = e
            if attempt < 4:
                print(f"retry {attempt + 1} after 503")
                time.sleep(backoff_seconds[attempt])
            else:
                raise last_error


def load_meta_exposures(
    client: bigquery.Client,
    project_id: str,
    dataset_meta: str,
    output_root: Path,
) -> dict[str, int]:
    folder = output_root / "meta_exposures"
    if not folder.is_dir():
        return {}
    impressions_id = _table_ref(project_id, dataset_meta, "impressions")
    clicks_id = _table_ref(project_id, dataset_meta, "clicks")
    max_date = _get_max_date(client, impressions_id, "date")

    paths_to_load = []
    for path in sorted(folder.glob("*.parquet")):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            file_date = None
        if max_date is not None and file_date is not None and file_date <= max_date:
            continue
        paths_to_load.append(path)

    if not paths_to_load:
        return {"impressions": 0, "clicks": 0}

    skipped = len(list(folder.glob("*.parquet"))) - len(paths_to_load)
    if skipped > 0:
        print(f"Skipping {skipped} file(s) already loaded (meta_exposures)")

    dfs = [pd.read_parquet(p) for p in paths_to_load]
    combined = pd.concat(dfs, ignore_index=True)
    if combined.empty:
        return {"impressions": 0, "clicks": 0}

    create_impressions = not _table_exists(client, impressions_id)
    create_clicks = not _table_exists(client, clicks_id)
    n_imp = _load_df_append(client, combined, impressions_id, create_impressions)
    clicked = combined.loc[combined["clicked"] == 1]
    n_clk = _load_df_append(client, clicked, clicks_id, create_clicks) if not clicked.empty else 0
    return {"impressions": n_imp, "clicks": n_clk}


def load_folder(
    client: bigquery.Client,
    project_id: str,
    dataset_meta: str,
    dataset_shopify: str,
    folder_name: str,
    output_root: Path,
) -> int:
    folder = output_root / folder_name
    if not folder.is_dir():
        return 0
    dataset_key, table_name = FOLDER_TO_TABLE[folder_name]
    dataset_id = dataset_meta if dataset_key == "meta" else dataset_shopify
    table_id = _table_ref(project_id, dataset_id, table_name)
    date_expr = FOLDER_MAX_DATE_EXPR.get(folder_name)
    max_date = (
        _get_max_date(client, table_id, date_expr) if date_expr else None
    )

    paths_to_load = []
    for path in sorted(folder.glob("*.parquet")):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            file_date = None
        if max_date is not None and file_date is not None and file_date <= max_date:
            continue
        paths_to_load.append(path)

    if not paths_to_load:
        return 0

    all_paths = list(folder.glob("*.parquet"))
    skipped = len(all_paths) - len(paths_to_load)
    if skipped > 0:
        print(f"Skipping {skipped} file(s) already loaded ({folder_name})")

    dfs = [pd.read_parquet(p) for p in paths_to_load]
    combined = pd.concat(dfs, ignore_index=True)
    if combined.empty:
        return 0

    create = not _table_exists(client, table_id)
    return _load_df_append(client, combined, table_id, create)


def run(config_path: Path | None = None) -> None:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config = _load_config(config_path)
    project_id = config["project_id"]
    dataset_meta = config["dataset_meta"]
    dataset_shopify = config["dataset_shopify"]

    client = bigquery.Client(project=project_id)

    # Meta exposures -> impressions + clicks
    meta_totals = load_meta_exposures(client, project_id, dataset_meta, OUTPUT_ROOT)
    for table, count in meta_totals.items():
        print(f"Loaded {count} rows into {dataset_meta}.{table}")

    # Other folders -> one table each
    for folder_name in ("orders", "line_items", "transactions", "refunds"):
        total = load_folder(
            client, project_id, dataset_meta, dataset_shopify, folder_name, OUTPUT_ROOT
        )
        _, table_name = FOLDER_TO_TABLE[folder_name]
        print(f"Loaded {total} rows into {dataset_shopify}.{table_name}")

    # Optional Meta entity folders (ads, adsets, campaigns, creatives)
    for folder_name in ("ads", "adsets", "campaigns", "creatives"):
        total = load_folder(
            client, project_id, dataset_meta, dataset_shopify, folder_name, OUTPUT_ROOT
        )
        if total > 0:
            _, table_name = FOLDER_TO_TABLE[folder_name]
            print(f"Loaded {total} rows into {dataset_meta}.{table_name}")


if __name__ == "__main__":
    run()