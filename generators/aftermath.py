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
    brand_state=None,
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
    - If refund: one row per refund with id, order_id, user_id, created_at,
      total_refunded_amount, currency (from order). NULL columns added
      for processed_at, transactions, etc.

    Parameters
    ----------
    orders_df : pandas.DataFrame
        Must have: order_id, customer_id, order_timestamp. Should have currency
        (refund.currency is copied from the order).
    line_items_df : pandas.DataFrame
        Must have: order_id, product_id, quantity, price.
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
        One row per refund. Shopify/dbt columns: id, order_id, user_id,
        created_at, total_refunded_amount, currency (from order); NULL:
        processed_at, updated_at, transactions, refund_line_items, restock,
        restock_type, note, admin_graphql_api_id.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    base_rate = config.get("base_refund_rate") or config.get("refund_rate", 0.07)
    base_rate = float(base_rate)

    # Upgrade 1: Brand lifecycle phase effect on refund rate
    if brand_state is not None:
        base_rate *= brand_state.get_phase_effects().get("refund_mult", 1.0)

    _empty_refund_columns = [
        "id", "order_id", "user_id", "created_at", "processed_at", "updated_at",
        "currency", "total_refunded_amount", "transactions", "refund_line_items",
        "restock", "restock_type", "note", "admin_graphql_api_id",
        "attributed_ad_id", "attributed_adset_id", "attributed_campaign_id", "attributed_creative_id",
    ]

    if orders_df.empty:
        return pd.DataFrame(columns=_empty_refund_columns)

    # Merge orders with customer traits (incl. trust_score Tier 2; recent_negative_velocity Tier 5; discount_dependency price)
    trait_cols = ["customer_id", "regret_propensity", "quality_expectation"]
    if "trust_score" in customers_df.columns:
        trait_cols.append("trust_score")
    if "recent_negative_velocity" in customers_df.columns:
        trait_cols.append("recent_negative_velocity")
    if "discount_dependency" in customers_df.columns:
        trait_cols.append("discount_dependency")
    orders = orders_df.merge(
        customers_df[trait_cols],
        on="customer_id",
        how="left",
    )

    # Per-order quality risk: join line_items → products (line_items already has product_id), then
    # take worst quality in the order (low → 1, mid → 0.3, high → 0)
    li = line_items_df.merge(
        products_df[["product_id", "quality_level"]], on="product_id", how="left"
    )
    li["quality_risk"] = li["quality_level"].map(_QUALITY_TO_RISK).fillna(0.5)
    order_quality_risk = li.groupby("order_id")["quality_risk"].max().reindex(orders_df["order_id"]).fillna(0.5)

    # Refund probability: base + regret + expectation + quality + trust (Tier 2) + fragility (Tier 5) + noise
    regret_lift = 0.08 * orders["regret_propensity"].values
    expectation_lift = 0.08 * orders["quality_expectation"].values
    quality_lift = 0.12 * order_quality_risk.values
    trust_weight = float(config.get("refund_trust_weight", 0.1))
    if "trust_score" in orders.columns:
        trust_penalty = (1.0 - orders["trust_score"].fillna(0.5).values) * trust_weight
    else:
        trust_penalty = 0.0
    # Tier 5: early disappointment (recent_negative_velocity) amplifies future refund probability
    fragility_weight = float(config.get("refund_fragility_weight", 0.12))
    if "recent_negative_velocity" in orders.columns:
        fragility_lift = orders["recent_negative_velocity"].fillna(0).values * fragility_weight
    else:
        fragility_lift = 0.0
    # Price psychology: discount dependency increases refund (promo-trained more likely to regret)
    dependency_refund_weight = float(config.get("dependency_refund_weight", 0.05))
    if "discount_dependency" in orders.columns:
        dependency_lift = orders["discount_dependency"].fillna(0).values * dependency_refund_weight
    else:
        dependency_lift = 0.0
    noise = rng.normal(0, 0.02, size=len(orders))
    refund_prob = (
        base_rate + regret_lift + expectation_lift + quality_lift + trust_penalty
        + fragility_lift + dependency_lift + noise
    )
    refund_prob = np.clip(refund_prob, 0.0, 1.0)

    # Decide refund per order
    refunded = rng.random(size=len(orders)) < refund_prob
    refunded_orders = orders.loc[refunded].copy()

    if refunded_orders.empty:
        return pd.DataFrame(columns=_empty_refund_columns)

    # Refund amount per order: sum(price * quantity) over line items
    line_items_df = line_items_df.copy()
    line_items_df["line_total"] = line_items_df["price"] * line_items_df["quantity"]
    order_totals = line_items_df.groupby("order_id")["line_total"].sum()
    refunded_orders["total_refunded_amount"] = refunded_orders["order_id"].map(order_totals).fillna(0.0)

    # Upgrade 3: Bimodal refund timing (delivery disappointment + usage disappointment)
    order_ts = pd.to_datetime(refunded_orders["order_timestamp"])
    n_refunds = len(refunded_orders)
    lag_cfg = config.get("lag_structure", {})
    refund_modes = lag_cfg.get("refund_timing_modes", [[5, 2, 0.6], [18, 5, 0.4]]) if isinstance(lag_cfg, dict) else [[5, 2, 0.6], [18, 5, 0.4]]
    if len(refund_modes) >= 2:
        m1_mean, m1_std, m1_weight = refund_modes[0]
        m2_mean, m2_std, _ = refund_modes[1]
        mode_choice = rng.random(size=n_refunds) < float(m1_weight)
        days_delta = np.where(
            mode_choice,
            rng.normal(float(m1_mean), float(m1_std), size=n_refunds),
            rng.normal(float(m2_mean), float(m2_std), size=n_refunds),
        )
        days_delta = np.clip(days_delta, 1, 45).astype(int)
    else:
        days_delta = rng.integers(3, 31, size=n_refunds)
    refunded_orders["created_at"] = order_ts + pd.to_timedelta(days_delta, unit="D")

    # Attribution: inherit from order (orders use last_attributed_*; refund output keeps attributed_*)
    order_attr_cols = ["last_attributed_ad_id", "last_attributed_adset_id", "last_attributed_campaign_id", "last_attributed_creative_id"]
    refund_attr_cols = ["attributed_ad_id", "attributed_adset_id", "attributed_campaign_id", "attributed_creative_id"]
    has_attr = [c for c in order_attr_cols if c in refunded_orders.columns]

    rows = []
    for _, row in refunded_orders.iterrows():
        r = {
            "id": str(uuid.uuid4()),
            "order_id": row["order_id"],
            "user_id": row["customer_id"],
            "created_at": row["created_at"],
            "total_refunded_amount": round(row["total_refunded_amount"], 2),
            "currency": row.get("currency"),  # from order (shopify_orders has currency)
        }
        for order_c in has_attr:
            r[refund_attr_cols[order_attr_cols.index(order_c)]] = row.get(order_c)
        rows.append(r)

    df = pd.DataFrame(rows)

    # Ensure attribution columns exist (if orders had none) → STRING null
    for c in refund_attr_cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype=object)

    # Typed nulls per contract: TIMESTAMP → pd.NaT, STRING → object (None), BOOL → nullable bool
    df["processed_at"] = pd.NaT
    df["updated_at"] = df["created_at"].copy()
    df["transactions"] = None  # STRING (object)
    df["refund_line_items"] = None  # STRING (object)
    df["restock"] = pd.array([pd.NA] * len(df), dtype="boolean")
    df["restock_type"] = None  # STRING (object)
    df["note"] = None  # STRING (object)
    df["admin_graphql_api_id"] = None  # STRING (object)

    return df
