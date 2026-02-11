"""
Post-purchase aftermath: refund simulation.

Refunds are modeled as emotional rupture events—disappointment after the
purchase that feeds into trust and erosion logic downstream. This module
keeps the mechanics simple (probability per order, single full-order refund);
smarter partial refunds and item-level logic can be added later.
"""

import uuid
import numpy as np
import pandas as pd


# Map product quality_level to a numeric "disappointment" factor (low → high risk)
_QUALITY_TO_RISK = {"low": 1.0, "mid": 0.3, "high": 0.0}


def simulate_refunds(
    orders_df: pd.DataFrame,
    line_items_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Model post-purchase disappointment by simulating which orders are refunded.

    Refunds are emotional rupture events: they feed trust erosion and
    attrition logic. Probability increases when the customer is prone to
    regret, has high quality expectations, or the product quality is low.

    Mechanics
    ---------
    - For each order, compute a refund probability.
    - Base rate from config (e.g. base_refund_rate or refund_rate, ~5–10%).
    - Increase when:
      - regret_propensity is high (customer more likely to regret)
      - quality_expectation is high (expectations harder to meet)
      - product quality_level is low (reality falls short)
    - Add small noise and clamp to [0, 1].
    - Bernoulli draw to decide refund.
    - If refund: one row per refund with refund_id, order_id, customer_id,
      refund_timestamp (order_timestamp + random 3–30 days), refund_amount
      (sum of price × quantity for that order).

    Parameters
    ----------
    orders_df : pandas.DataFrame
        Must have: order_id, customer_id, order_timestamp.
    line_items_df : pandas.DataFrame
        Must have: order_id, variant_id, quantity, price.
    customers_df : pandas.DataFrame
        Must have: customer_id, regret_propensity, quality_expectation.
    products_df : pandas.DataFrame
        Must have: product_id, quality_level.
    variants_df : pandas.DataFrame
        Must have: variant_id, product_id.
    config : dict
        Optional: base_refund_rate or refund_rate (float, e.g. 0.05–0.10).
        Optional: seed (int) for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per refund: refund_id, order_id, customer_id,
        refund_timestamp, refund_amount.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    base_rate = config.get("base_refund_rate") or config.get("refund_rate", 0.07)
    base_rate = float(base_rate)

    if orders_df.empty:
        return pd.DataFrame(
            columns=[
                "refund_id",
                "order_id",
                "customer_id",
                "refund_timestamp",
                "refund_amount",
            ]
        )

    # Merge orders with customer traits
    orders = orders_df.merge(
        customers_df[["customer_id", "regret_propensity", "quality_expectation"]],
        on="customer_id",
        how="left",
    )

    # Per-order quality risk: join line_items → variants → products, then
    # take worst quality in the order (low → 1, mid → 0.3, high → 0)
    li = line_items_df.merge(
        variants_df[["variant_id", "product_id"]], on="variant_id", how="left"
    )
    li = li.merge(
        products_df[["product_id", "quality_level"]], on="product_id", how="left"
    )
    li["quality_risk"] = li["quality_level"].map(_QUALITY_TO_RISK).fillna(0.5)
    order_quality_risk = li.groupby("order_id")["quality_risk"].max().reindex(orders_df["order_id"]).fillna(0.5)

    # Refund probability: base + regret + expectation + quality disappointment + noise
    regret_lift = 0.08 * orders["regret_propensity"].values
    expectation_lift = 0.08 * orders["quality_expectation"].values
    quality_lift = 0.12 * order_quality_risk.values
    noise = rng.normal(0, 0.02, size=len(orders))
    refund_prob = base_rate + regret_lift + expectation_lift + quality_lift + noise
    refund_prob = np.clip(refund_prob, 0.0, 1.0)

    # Decide refund per order
    refunded = rng.random(size=len(orders)) < refund_prob
    refunded_orders = orders.loc[refunded].copy()

    if refunded_orders.empty:
        return pd.DataFrame(
            columns=[
                "refund_id",
                "order_id",
                "customer_id",
                "refund_timestamp",
                "refund_amount",
            ]
        )

    # Refund amount per order: sum(price * quantity) over line items
    line_items_df = line_items_df.copy()
    line_items_df["line_total"] = line_items_df["price"] * line_items_df["quantity"]
    order_totals = line_items_df.groupby("order_id")["line_total"].sum()
    refunded_orders["refund_amount"] = refunded_orders["order_id"].map(order_totals).fillna(0.0)

    # Refund timestamp: order_timestamp + random 3–30 days
    order_ts = pd.to_datetime(refunded_orders["order_timestamp"])
    days_delta = rng.integers(3, 31, size=len(refunded_orders))
    refunded_orders["refund_timestamp"] = order_ts + pd.to_timedelta(days_delta, unit="D")

    rows = []
    for _, row in refunded_orders.iterrows():
        rows.append({
            "refund_id": str(uuid.uuid4()),
            "order_id": row["order_id"],
            "customer_id": row["customer_id"],
            "refund_timestamp": row["refund_timestamp"],
            "refund_amount": round(row["refund_amount"], 2),
        })

    return pd.DataFrame(rows)