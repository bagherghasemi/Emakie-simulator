"""
Post-purchase operations: fulfillment (shipping) simulation.

Fulfillments represent the physical shipping of an order. Not every order ships;
shipping and delivery delays depend on product shipping_difficulty (easy/normal/hard).
"""

import uuid
import numpy as np
import pandas as pd


# Shipping difficulty → (min_days, max_days) for time from order to ship
_DIFFICULTY_SHIP_DAYS = {
    "easy": (1, 2),
    "normal": (2, 4),
    "hard": (4, 8),
}
# Delivery: 1–5 days after ship
_DELIVERY_DAYS_MIN, _DELIVERY_DAYS_MAX = 1, 5


def simulate_fulfillments(
    orders_df: pd.DataFrame,
    line_items_df: pd.DataFrame,
    products_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Simulate which orders are fulfilled (shipped) and when they ship and deliver.

    Rules:
    - Not every order ships; base ship probability = 0.95.
    - Shipping delay depends on product.shipping_difficulty (per order: MAX over
      line_items → products): easy 1–2 days, normal 2–4, hard 4–8.
    - Delivery happens 1–5 days after shipping.
    - Status: "delivered" if delivered_at <= simulation end_date, else "shipped".

    Parameters
    ----------
    orders_df : pandas.DataFrame
        Must have: order_id, order_timestamp.
    line_items_df : pandas.DataFrame
        Must have: order_id, product_id.
    products_df : pandas.DataFrame
        Must have: product_id, shipping_difficulty.
    variants_df : pandas.DataFrame
        Unused; kept for API compatibility.
    config : dict
        Optional: simulation.end_date (str YYYY-MM-DD) for status.
        Optional: base_fulfillment_rate or ship_probability (float, default 0.95).
        Optional: seed (int) for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per fulfillment. Shopify/dbt columns: id, order_id, status,
        created_at; NULL: location_id, shipment_status, updated_at,
        tracking_company, tracking_number, tracking_numbers, tracking_url,
        tracking_urls, service, name, receipt, line_items, admin_graphql_api_id.
        Extra (for loader idempotency): shipped_at, delivered_at.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    base_ship_prob = config.get("base_fulfillment_rate") or config.get(
        "ship_probability", 0.95
    )
    base_ship_prob = float(base_ship_prob)

    sim = config.get("simulation") or {}
    end_str = sim.get("end_date", "2099-12-31")
    simulation_end = pd.Timestamp(end_str)

    _empty_fulfillment_columns = [
        "id", "order_id", "location_id", "status", "shipment_status",
        "created_at", "updated_at", "tracking_company", "tracking_number",
        "tracking_numbers", "tracking_url", "tracking_urls", "service", "name",
        "receipt", "line_items", "admin_graphql_api_id", "shipped_at", "delivered_at",
    ]

    if orders_df.empty:
        return pd.DataFrame(columns=_empty_fulfillment_columns)

    # Per-order max shipping_difficulty: line_items → products (line_items already has product_id)
    li = line_items_df.merge(
        products_df[["product_id", "shipping_difficulty"]],
        on="product_id",
        how="left",
    )
    # Rank: easy < normal < hard → take hardest in order
    diff_rank = {"easy": 0, "normal": 1, "hard": 2}
    li["_diff_rank"] = li["shipping_difficulty"].map(diff_rank).fillna(1)
    order_max_rank = li.groupby("order_id")["_diff_rank"].max()
    order_difficulty = order_max_rank.map(
        {0: "easy", 1: "normal", 2: "hard"}
    ).reindex(orders_df["order_id"]).fillna("normal")

    # Which orders ship (Bernoulli)
    ship_roll = rng.random(size=len(orders_df))
    ships = ship_roll < base_ship_prob
    shipped_orders = orders_df.loc[ships].copy()

    if shipped_orders.empty:
        return pd.DataFrame(columns=_empty_fulfillment_columns)

    # Shipping delay (days) by difficulty
    ship_delays = []
    for oid in shipped_orders["order_id"]:
        diff = order_difficulty.get(oid, "normal")
        lo, hi = _DIFFICULTY_SHIP_DAYS.get(diff, (2, 4))
        ship_delays.append(rng.integers(lo, hi + 1))
    shipped_orders["_ship_days"] = ship_delays

    order_ts = pd.to_datetime(shipped_orders["order_timestamp"])
    shipped_orders["created_at"] = order_ts
    shipped_orders["shipped_at"] = order_ts + pd.to_timedelta(
        shipped_orders["_ship_days"], unit="D"
    )

    delivery_days = rng.integers(
        _DELIVERY_DAYS_MIN, _DELIVERY_DAYS_MAX + 1, size=len(shipped_orders)
    )
    shipped_orders["delivered_at"] = shipped_orders["shipped_at"] + pd.to_timedelta(
        delivery_days, unit="D"
    )
    delivered_vals = shipped_orders["delivered_at"]
    if delivered_vals.dt.tz is not None:
        delivered_vals = delivered_vals.dt.tz_localize(None)
    shipped_orders["status"] = np.where(
        delivered_vals.le(simulation_end), "delivered", "shipped"
    )

    rows = []
    for _, row in shipped_orders.iterrows():
        rows.append({
            "id": str(uuid.uuid4()),
            "order_id": row["order_id"],
            "created_at": row["created_at"],
            "shipped_at": row["shipped_at"],
            "delivered_at": row["delivered_at"],
            "status": row["status"],
        })

    df = pd.DataFrame(rows)

    # Typed nulls per contract: TIMESTAMP → pd.NaT, STRING → None
    df["updated_at"] = df["created_at"].copy()  # TIMESTAMP (same as created_at for fulfillments)
    for col in [
        "location_id", "shipment_status",
        "tracking_company", "tracking_number", "tracking_numbers",
        "tracking_url", "tracking_urls", "service", "name",
        "receipt", "line_items", "admin_graphql_api_id",
    ]:
        df[col] = None  # STRING (object)

    # Column order: dbt-compatible first, then shipped_at/delivered_at for loader
    df = df[
        [
            "id", "order_id", "location_id", "status", "shipment_status",
            "created_at", "updated_at", "tracking_company", "tracking_number",
            "tracking_numbers", "tracking_url", "tracking_urls", "service", "name",
            "receipt", "line_items", "admin_graphql_api_id",
            "shipped_at", "delivered_at",
        ]
    ]

    return df
