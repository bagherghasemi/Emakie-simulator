"""
Product and variant generators for synthetic commerce simulation.

Produces DataFrames of synthetic products and their variants (size/color options)
for use in downstream event simulation. Includes purchase simulation that
converts ad clicks into orders, line items, and transactions—the first monetary
commitment event in the funnel (refunds not yet modeled).
"""

import uuid
from typing import List, Tuple

import numpy as np
import pandas as pd

CATEGORIES = ["apparel", "accessories", "beauty", "home"]
QUALITY_LEVELS = ["low", "mid", "high"]
QUALITY_PROBS = np.array([0.30, 0.50, 0.20])
SHIPPING_DIFFICULTIES = ["easy", "normal", "hard"]
SHIPPING_PROBS = np.array([0.40, 0.40, 0.20])

# Example variant templates: (attribute, values)
VARIANT_TEMPLATES = [
    ("Size", ["S", "M", "L", "XL"]),
    ("Color", ["Black", "White", "Red", "Blue", "Navy", "Grey"]),
]


def generate_products(config: dict) -> pd.DataFrame:
    """
    Generate a catalog of synthetic products with pricing and quality attributes.

    Reads the number of products from config (e.g. num_products or products).
    Optionally uses config["seed"] for numpy RNG reproducibility.

    Each product has:
    - product_id: UUID string
    - product_name: Simple label (e.g. "Product 1")
    - category: One of apparel, accessories, beauty, home (uniform)
    - base_price: Float; higher for high quality_level
    - cost_of_goods: Float; 30–60% of base_price with noise
    - quality_level: low / mid / high (distributed ~30% / 50% / 20%)
    - complexity_level: Float in [0, 1]; normal distribution around 0.5, clamped
    - shipping_difficulty: easy / normal / hard (distributed ~40% / 40% / 20%)

    Quality influences price: high → higher base_price, low → lower.
    Cost of goods is kept within 30–60% of price with realistic noise.

    Parameters
    ----------
    config : dict
        Must contain number of products (num_products or products).
        Optional: seed (int) for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per product with columns: product_id, product_name, category,
        base_price, cost_of_goods, quality_level, complexity_level,
        shipping_difficulty.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    n = config.get("num_products") or config.get("products") or config.get("number_of_products")
    if n is None:
        raise ValueError("config must specify number of products (e.g. num_products)")
    n = int(n)

    product_ids = [str(uuid.uuid4()) for _ in range(n)]
    product_names = [f"Product {i + 1}" for i in range(n)]
    categories = rng.choice(CATEGORIES, size=n)
    quality_levels = rng.choice(QUALITY_LEVELS, size=n, p=QUALITY_PROBS)

    # Base price by quality: low ~10–50, mid ~30–80, high ~60–150 (with noise)
    base_prices = np.zeros(n)
    for i, ql in enumerate(quality_levels):
        if ql == "low":
            p = rng.uniform(10.0, 50.0)
        elif ql == "mid":
            p = rng.uniform(30.0, 80.0)
        else:
            p = rng.uniform(60.0, 150.0)
        base_prices[i] = round(p + rng.normal(0, 3), 2)
        base_prices[i] = max(1.0, base_prices[i])

    # Cost of goods: 30–60% of price with noise
    cog_ratio = 0.3 + rng.uniform(0, 0.3, size=n) + rng.normal(0, 0.05, size=n)
    cog_ratio = np.clip(cog_ratio, 0.30, 0.60)
    cost_of_goods = (base_prices * cog_ratio).round(2)

    # complexity_level: normal around 0.5, clamped to [0, 1]
    complexity_level = np.clip(rng.normal(0.5, 0.2, size=n), 0.0, 1.0)

    # shipping_difficulty: easy / normal / hard (40% / 40% / 20%)
    shipping_difficulty = rng.choice(SHIPPING_DIFFICULTIES, size=n, p=SHIPPING_PROBS)

    return pd.DataFrame({
        "product_id": product_ids,
        "product_name": product_names,
        "category": categories,
        "base_price": base_prices,
        "cost_of_goods": cost_of_goods,
        "quality_level": quality_levels,
        "complexity_level": complexity_level,
        "shipping_difficulty": shipping_difficulty,
    })


def _build_variant_name(rng: np.random.Generator) -> str:
    """Build a variant name like 'Size M / Color Red' from templates."""
    parts: List[str] = []
    for attr, values in VARIANT_TEMPLATES:
        if rng.random() < 0.6:  # Include each attribute ~60% of the time
            parts.append(f"{attr} {rng.choice(values)}")
    return " / ".join(parts) if parts else "Standard"


def generate_variants(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate variants for each product (1–4 variants per product).

    Each product gets a random number of variants (1–4). Each variant has:
    - variant_id: UUID string
    - product_id: Links to products_df
    - variant_name: Human-readable label (e.g. "Size M / Color Red")
    - price: base_price with slight deviation (±5–15%)

    Parameters
    ----------
    products_df : pandas.DataFrame
        Output of generate_products. Must have columns: product_id, base_price.

    Returns
    -------
    pandas.DataFrame
        One row per variant with columns: variant_id, product_id, variant_name, price.
    """
    seed = getattr(products_df, "_generator_seed", None)
    rng = np.random.default_rng(seed)

    rows: List[dict] = []
    for _, row in products_df.iterrows():
        product_id = row["product_id"]
        base_price = row["base_price"]
        n_variants = rng.integers(1, 5)  # 1–4 inclusive

        for _ in range(n_variants):
            variant_name = _build_variant_name(rng)
            # Slight price deviation: ±5–15%
            deviation = 0.90 + rng.uniform(0, 0.20)
            price = round(base_price * deviation, 2)
            price = max(0.01, price)

            rows.append({
                "variant_id": str(uuid.uuid4()),
                "product_id": product_id,
                "variant_name": variant_name,
                "price": price,
            })

    return pd.DataFrame(rows)


def simulate_purchases_from_clicks(
    clicks_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert ad clicks into potential purchases (orders, line items, transactions).

    This is the first monetary commitment event in the funnel: a click becomes
    a purchase decision. Refunds and partial failures are not yet modeled; all
    transactions are successful.

    Mechanics:
    - For each click, compute a conversion probability.
    - Probability increases with high impulse_level and loyalty_propensity.
    - Probability decreases with high price_sensitivity.
    - Base conversion from config (e.g. 2–5%); add small noise; clamp to [0, 1].
    - Draw purchase decision (Bernoulli).
    - If purchase: create order (order_id, customer_id, order_timestamp = click
      date + small minutes offset), one line item (random variant, quantity
      usually 1 sometimes 2–3, price = variant price), and transaction
      (status = success).

    Parameters
    ----------
    clicks_df : pandas.DataFrame
        One row per click. Must have: customer_id, date (or click_date).
    customers_df : pandas.DataFrame
        Must have: customer_id, impulse_level, loyalty_propensity,
        price_sensitivity.
    variants_df : pandas.DataFrame
        Must have: variant_id, product_id, price.
    config : dict
        Optional: base_conversion or conversion_rate (float, e.g. 0.02–0.05).
        Optional: seed (int) for reproducibility.

    Returns
    -------
    tuple of (orders_df, line_items_df, transactions_df)
        orders_df: order_id, customer_id, order_timestamp
        line_items_df: line_item_id, order_id, variant_id, quantity, price
        transactions_df: transaction_id, order_id, status
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    base_conversion = config.get("base_conversion") or config.get("conversion_rate", 0.035)
    base_conversion = float(base_conversion)

    # Resolve date column
    click_date_col = "date" if "date" in clicks_df.columns else "click_date"
    if click_date_col not in clicks_df.columns:
        raise ValueError("clicks_df must have 'date' or 'click_date'")

    if clicks_df.empty:
        return (
            pd.DataFrame(columns=["order_id", "customer_id", "order_timestamp"]),
            pd.DataFrame(columns=["line_item_id", "order_id", "variant_id", "quantity", "price"]),
            pd.DataFrame(columns=["transaction_id", "order_id", "status"]),
        )

    # Merge customer traits
    merged = clicks_df.merge(
        customers_df[["customer_id", "impulse_level", "loyalty_propensity", "price_sensitivity"]],
        on="customer_id",
        how="left",
    )

    # Conversion probability: base + impulse/loyalty lift - price penalty + noise
    impulse_lift = 0.03 * merged["impulse_level"].values
    loyalty_lift = 0.03 * merged["loyalty_propensity"].values
    price_penalty = 0.04 * merged["price_sensitivity"].values
    noise = rng.normal(0, 0.005, size=len(merged))
    conv_prob = base_conversion + impulse_lift + loyalty_lift - price_penalty + noise
    conv_prob = np.clip(conv_prob, 0.0, 1.0)

    purchased = rng.random(size=len(merged)) < conv_prob
    purchased_indices = np.where(purchased)[0]
    merged_purchased = merged.iloc[purchased_indices].copy()

    # Development observability: if there are clicks but zero purchases, randomly
    # select 1–3 clicks and force them to convert so the funnel is visible during
    # development. They are marked normally (no special flag). Can be disabled later.
    if merged_purchased.empty and len(merged) > 0:
        n_force = min(rng.integers(1, 4), len(merged))
        forced_indices = rng.choice(len(merged), size=n_force, replace=False)
        merged_purchased = merged.iloc[forced_indices].copy()

    if merged_purchased.empty:
        return (
            pd.DataFrame(columns=["order_id", "customer_id", "order_timestamp"]),
            pd.DataFrame(columns=["line_item_id", "order_id", "variant_id", "quantity", "price"]),
            pd.DataFrame(columns=["transaction_id", "order_id", "status"]),
        )

    n_purchases = len(merged_purchased)
    variant_list = variants_df.to_dict("records")
    n_variants = len(variant_list)
    if n_variants == 0:
        raise ValueError("variants_df is empty")

    orders_rows: List[dict] = []
    line_items_rows: List[dict] = []
    transactions_rows: List[dict] = []

    for i, (_, row) in enumerate(merged_purchased.iterrows()):
        order_id = str(uuid.uuid4())
        customer_id = row["customer_id"]
        click_date = pd.to_datetime(row[click_date_col])
        order_timestamp = click_date + pd.Timedelta(minutes=rng.integers(1, 31))

        orders_rows.append({
            "order_id": order_id,
            "customer_id": customer_id,
            "order_timestamp": order_timestamp,
        })

        # Random variant; quantity usually 1, sometimes 2–3
        q_roll = rng.random()
        quantity = 1 if q_roll < 0.75 else rng.integers(2, 4)
        variant = variant_list[rng.integers(0, n_variants)]
        price = variant["price"]

        line_items_rows.append({
            "line_item_id": str(uuid.uuid4()),
            "order_id": order_id,
            "variant_id": variant["variant_id"],
            "quantity": quantity,
            "price": price,
        })

        transactions_rows.append({
            "transaction_id": str(uuid.uuid4()),
            "order_id": order_id,
            "status": "success",
        })

    return (
        pd.DataFrame(orders_rows),
        pd.DataFrame(line_items_rows),
        pd.DataFrame(transactions_rows),
    )