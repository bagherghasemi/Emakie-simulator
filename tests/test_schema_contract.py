"""
CI protection: run minimal simulation and validate output dtypes.
If any generator emits wrong dtype (e.g. updated_at as object) → test fails.
"""
import sys
from pathlib import Path

import pytest

# Ensure repo root on path when running as script
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Bootstrap emakie_simulator when run as pytest from repo root
if "emakie_simulator" not in sys.modules:
    import types
    _pkg = types.ModuleType("emakie_simulator")
    _pkg.__path__ = [str(REPO_ROOT)]
    sys.modules["emakie_simulator"] = _pkg

from generators.commerce import (
    generate_products,
    generate_variants,
    simulate_purchases_from_clicks,
    simulate_repeat_purchases_for_day,
)
from generators.humans import generate_customers
from generators.aftermath import simulate_refunds
from generators.operations import simulate_fulfillments
from generators.meta import generate_ad_accounts, generate_campaigns, generate_adsets, generate_ads, generate_creatives
from generators.meta_reporting import build_ad_performance_daily
from loaders import schema_contract


def _minimal_config():
    return {
        "seed": 99,
        "simulation": {"start_date": "2023-06-01", "end_date": "2023-06-02"},
        "num_customers": 5,
        "num_products": 3,
        "creatives": 2,
        "currency": "USD",
    }


@pytest.fixture
def minimal_config():
    return _minimal_config()


def test_refunds_updated_at_is_datetime(minimal_config):
    """Regression: refunds.updated_at must be datetime64, not object (was STRING in BQ)."""
    config = minimal_config
    products_df = generate_products(config)
    variants_df = generate_variants(products_df)
    customers_df = generate_customers(variants_df, config)
    clicks_df = __make_minimal_clicks(customers_df)
    orders_df, line_items_df, transactions_df, _, _, _, _ = simulate_purchases_from_clicks(
        clicks_df, customers_df, variants_df, products_df, config
    )
    refunds_df = simulate_refunds(orders_df, line_items_df, customers_df, products_df, variants_df, config)
    if refunds_df.empty:
        pytest.skip("no refunds in minimal run")
    assert str(refunds_df["updated_at"].dtype).startswith("datetime64"), (
        f"refunds.updated_at must be datetime64, got {refunds_df['updated_at'].dtype}"
    )
    assert str(refunds_df["processed_at"].dtype).startswith("datetime64"), (
        f"refunds.processed_at must be datetime64, got {refunds_df['processed_at'].dtype}"
    )


def test_fulfillments_updated_at_is_datetime(minimal_config):
    """Fulfillments.updated_at must be datetime64."""
    config = minimal_config
    products_df = generate_products(config)
    variants_df = generate_variants(products_df)
    customers_df = generate_customers(variants_df, config)
    clicks_df = __make_minimal_clicks(customers_df)
    orders_df, line_items_df, transactions_df, _, _, _, _ = simulate_purchases_from_clicks(
        clicks_df, customers_df, variants_df, products_df, config
    )
    fulfillments_df = simulate_fulfillments(orders_df, line_items_df, products_df, variants_df, config)
    if fulfillments_df.empty:
        pytest.skip("no fulfillments in minimal run")
    assert str(fulfillments_df["updated_at"].dtype).startswith("datetime64"), (
        f"fulfillments.updated_at must be datetime64, got {fulfillments_df['updated_at'].dtype}"
    )


def test_orders_cancelled_closed_are_datetime(minimal_config):
    """Orders optional timestamp columns must be datetime64."""
    config = minimal_config
    products_df = generate_products(config)
    variants_df = generate_variants(products_df)
    customers_df = generate_customers(variants_df, config)
    clicks_df = __make_minimal_clicks(customers_df)
    orders_df, _, _, _, _, _, _ = simulate_purchases_from_clicks(
        clicks_df, customers_df, variants_df, products_df, config
    )
    assert "cancelled_at" in orders_df.columns and (
        str(orders_df["cancelled_at"].dtype).startswith("datetime64")
    ), f"orders.cancelled_at must be datetime64, got {orders_df['cancelled_at'].dtype}"
    assert str(orders_df["closed_at"].dtype).startswith("datetime64"), (
        f"orders.closed_at must be datetime64, got {orders_df['closed_at'].dtype}"
    )


def test_transactions_test_is_boolean(minimal_config):
    """Transactions.test must be boolean (nullable)."""
    config = minimal_config
    products_df = generate_products(config)
    variants_df = generate_variants(products_df)
    customers_df = generate_customers(variants_df, config)
    clicks_df = __make_minimal_clicks(customers_df)
    _, _, transactions_df, _, _, _, _ = simulate_purchases_from_clicks(
        clicks_df, customers_df, variants_df, products_df, config
    )
    assert str(transactions_df["test"].dtype) in ("bool", "boolean"), (
        f"transactions.test must be bool/boolean, got {transactions_df['test'].dtype}"
    )


def test_validate_contract_when_contract_loaded(minimal_config):
    """When contract CSV is present, full validation must pass on generator outputs."""
    if not schema_contract.TABLE_CONTRACT:
        pytest.skip("schema contract not loaded (seeds/source_schema_contract.csv not found)")
    config = minimal_config
    products_df = generate_products(config)
    variants_df = generate_variants(products_df)
    customers_df = generate_customers(variants_df, config)
    clicks_df = __make_minimal_clicks(customers_df)
    orders_df, line_items_df, transactions_df, _, _, checkouts_df, _ = simulate_purchases_from_clicks(
        clicks_df, customers_df, variants_df, products_df, config
    )
    refunds_df = simulate_refunds(orders_df, line_items_df, customers_df, products_df, variants_df, config)
    fulfillments_df = simulate_fulfillments(orders_df, line_items_df, products_df, variants_df, config)

    schema_contract.validate_dataframe_against_contract(orders_df, "shopify_orders")
    schema_contract.validate_dataframe_against_contract(line_items_df, "shopify_order_line_items")
    schema_contract.validate_dataframe_against_contract(transactions_df, "shopify_transactions")
    schema_contract.validate_dataframe_against_contract(refunds_df, "shopify_refunds")
    schema_contract.validate_dataframe_against_contract(fulfillments_df, "shopify_fulfillments")
    schema_contract.validate_dataframe_against_contract(checkouts_df, "shopify_checkouts")
    schema_contract.validate_dataframe_against_contract(products_df, "shopify_products")
    schema_contract.validate_dataframe_against_contract(variants_df, "shopify_product_variants")
    schema_contract.validate_dataframe_against_contract(customers_df, "shopify_customers")


def __make_minimal_clicks(customers_df):
    import pandas as pd
    cids = customers_df["customer_id"].head(3).tolist()
    return pd.DataFrame({
        "customer_id": cids,
        "date": [pd.Timestamp("2023-06-01")] * len(cids),
        "ad_id": ["ad1"] * len(cids),
        "adset_id": ["aset1"] * len(cids),
        "campaign_id": ["camp1"] * len(cids),
        "creative_id": ["creat1"] * len(cids),
    })
