"""
Product and variant generators for synthetic commerce simulation.

Produces DataFrames of synthetic products and their variants (size/color options)
for use in downstream event simulation. Includes purchase simulation that
converts ad clicks into orders, line items, and transactions—the first monetary
commitment event in the funnel (refunds not yet modeled).
"""

import uuid
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .humans import FAKE_REGIONS
from .meta import (
    _ANGLE_PARAMS, _HOOK_PARAMS, _CREATIVE_TYPE_PARAMS,
    _PERSONA_TRAIT_AFFINITY, _persona_resonance, _is_video_type,
)
from .psychological_state import _trait_creative_alignment

# Geography catalog: same as humans.py uses for customer default addresses
ADDRESS_COUNTRIES = ["US", "CA", "UK", "DE"]

# Cities by country for billing_address_city (derived from billing_address_country)
CITY_BY_COUNTRY = {
    "US": ["New York", "Los Angeles", "Chicago", "Austin", "Miami"],
    "CA": ["Toronto", "Vancouver", "Montreal"],
    "UK": ["London", "Manchester", "Birmingham"],
    "DE": ["Berlin", "Munich", "Hamburg"],
}


def _pick_city_for_country(country: str, rng: np.random.Generator) -> str:
    """Pick a deterministic city from CITY_BY_COUNTRY. Never returns null."""
    cities = CITY_BY_COUNTRY.get(country, ["Unknown"])
    return cities[rng.integers(0, len(cities))]


def _enrich_order_addresses(
    default_country: object,
    default_region: object,
    rng: np.random.Generator,
) -> Tuple[str, str, str, str]:
    """
    Compute shipping and billing country/region from customer default address.
    Never returns nulls. Deterministic given RNG.
    Shipping: 90% same as default, 7% same country diff region, 3% diff country.
    Billing: 90% = shipping, 8% = default, 2% = another region (same country as shipping).
    """
    dc = "US" if default_country is None or (isinstance(default_country, float) and pd.isna(default_country)) else str(default_country).strip() or "US"
    dr = "Central" if default_region is None or (isinstance(default_region, float) and pd.isna(default_region)) else str(default_region).strip() or "Central"
    if dc not in ADDRESS_COUNTRIES:
        dc = "US"
    if dr not in FAKE_REGIONS:
        dr = "Central"

    # Shipping: 90% same, 7% same country diff region, 3% diff country
    s_roll = rng.random()
    if s_roll < 0.90:
        ship_country, ship_region = dc, dr
    elif s_roll < 0.97:
        other_regions = [r for r in FAKE_REGIONS if r != dr]
        ship_region = other_regions[rng.integers(0, len(other_regions))] if other_regions else FAKE_REGIONS[0]
        ship_country = dc
    else:
        other_countries = [c for c in ADDRESS_COUNTRIES if c != dc]
        ship_country = other_countries[rng.integers(0, len(other_countries))] if other_countries else ADDRESS_COUNTRIES[0]
        ship_region = FAKE_REGIONS[rng.integers(0, len(FAKE_REGIONS))]

    # Billing: 90% = shipping, 8% = default, 2% = another region (same country as shipping)
    b_roll = rng.random()
    if b_roll < 0.90:
        bill_country, bill_region = ship_country, ship_region
    elif b_roll < 0.98:
        bill_country, bill_region = dc, dr
    else:
        bill_country = ship_country
        other_regions = [r for r in FAKE_REGIONS if r != ship_region]
        bill_region = other_regions[rng.integers(0, len(other_regions))] if other_regions else FAKE_REGIONS[0]

    return (ship_country, ship_region, bill_country, bill_region)


CATEGORIES = ["apparel", "accessories", "beauty", "home"]
QUALITY_LEVELS = ["low", "mid", "high"]
QUALITY_PROBS = np.array([0.30, 0.50, 0.20])
SHIPPING_DIFFICULTIES = ["easy", "normal", "hard"]
SHIPPING_PROBS = np.array([0.40, 0.40, 0.20])

# Vendors for Shopify-style product output
VENDORS = ["Acme Co", "Brand One", "Style House", "Urban Goods", "Home & Co", "Beauty Lab", "Fresh Finds"]
# status: mostly active, small chance draft or archived
STATUS_OPTIONS = ["active", "active", "active", "active", "active", "draft", "archived"]
STATUS_PROBS = np.array([1.0 / len(STATUS_OPTIONS)] * len(STATUS_OPTIONS))

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

    Each product has Shopify-style columns: id, title, product_type, vendor,
    status, created_at, updated_at; simulator columns: base_price, cost_of_goods,
    quality_level, complexity_level, shipping_difficulty. product_id is present
    as alias of id for downstream.

    Parameters
    ----------
    config : dict
        Must contain number of products (num_products or products).
        Optional: simulation.start_date for created_at; seed (int) for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per product. Shopify columns: id, title, product_type, vendor,
        status, created_at, updated_at; NULL: handle, body_html, tags,
        published_at, published_scope, image_src, image_alt_text, template_suffix,
        admin_graphql_api_id. Simulator columns: base_price, cost_of_goods,
        quality_level, complexity_level, shipping_difficulty. product_id = id.
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

    # Shopify-style schema: rename and enrich
    df = pd.DataFrame({
        "id": product_ids,
        "title": product_names,
        "product_type": categories,
        "base_price": base_prices,
        "cost_of_goods": cost_of_goods,
        "quality_level": quality_levels,
        "complexity_level": complexity_level,
        "shipping_difficulty": shipping_difficulty,
    })

    df["vendor"] = rng.choice(VENDORS, size=n).tolist()
    df["status"] = rng.choice(STATUS_OPTIONS, size=n, p=STATUS_PROBS).tolist()

    # created_at: random timestamp before simulation start
    sim = config.get("simulation", config)
    start_date = pd.to_datetime(sim.get("start_date", "2023-01-01"))
    # Random timestamps in the year before start
    start_ts = (start_date - pd.Timedelta(days=365)).value
    end_ts = start_date.value
    created_ts = rng.integers(start_ts, end_ts + 1, size=n)
    df["created_at"] = pd.to_datetime(created_ts, unit="ns")
    df["updated_at"] = df["created_at"]

    # Fields present in Shopify but not simulated → typed nulls per contract
    for col in ["handle", "body_html", "tags", "published_scope", "image_src", "image_alt_text", "template_suffix", "admin_graphql_api_id"]:
        df[col] = None  # STRING (object)
    df["published_at"] = pd.NaT  # TIMESTAMP
    df["has_only_default_variant"] = pd.array([pd.NA] * len(df), dtype="boolean")  # BOOL

    # Alias for downstream (generate_variants, simulate_fulfillments, simulate_refunds)
    df["product_id"] = df["id"].values

    return df


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

    Each variant has Shopify-style columns: id, product_id, title, price, sku,
    inventory_quantity, requires_shipping, taxable, created_at, updated_at;
    NULL: barcode, option_1/2/3, compare_at_price, etc. variant_id = id for downstream.

    Parameters
    ----------
    products_df : pandas.DataFrame
        Output of generate_products. Must have columns: product_id, base_price.
        Optional: created_at (used for variant created_at).

    Returns
    -------
    pandas.DataFrame
        One row per variant. Shopify columns: id, product_id, title, price, sku,
        inventory_quantity, requires_shipping, taxable, created_at, updated_at;
        NULL for barcode, option_1/2/3, compare_at_price, etc. variant_id = id.
    """
    seed = getattr(products_df, "_generator_seed", None)
    rng = np.random.default_rng(seed)

    rows: List[dict] = []
    for _, row in products_df.iterrows():
        product_id = row["product_id"]
        base_price = row["base_price"]
        product_created_at = row.get("created_at", pd.Timestamp("2020-01-01"))
        n_variants = rng.integers(1, 5)  # 1–4 inclusive

        for j in range(n_variants):
            variant_id = str(uuid.uuid4())
            variant_name = _build_variant_name(rng)
            # Slight price deviation: ±5–15%
            deviation = 0.90 + rng.uniform(0, 0.20)
            price = round(base_price * deviation, 2)
            price = max(0.01, price)

            sku = f"SKU-{product_id[:8]}-{j}"
            rows.append({
                "id": variant_id,
                "product_id": product_id,
                "title": variant_name,
                "price": price,
                "sku": sku,
                "created_at": product_created_at,
            })

    df = pd.DataFrame(rows)

    df["updated_at"] = df["created_at"]
    df["inventory_quantity"] = rng.integers(0, 501, size=len(df))
    # requires_shipping: True for most (~90%)
    df["requires_shipping"] = rng.random(size=len(df)) < 0.90
    df["taxable"] = True

    # Fields present in Shopify but not simulated → typed nulls per contract
    for col in ["barcode", "option_1", "option_2", "option_3", "presentment_prices", "inventory_item_id", "inventory_management", "inventory_policy", "weight_unit", "admin_graphql_api_id", "image_id"]:
        df[col] = None  # STRING (object)
    df["compare_at_price"] = np.nan  # NUMERIC
    df["weight"] = np.nan  # NUMERIC
    df["available_for_sale"] = pd.array([pd.NA] * len(df), dtype="boolean")  # BOOL
    df["position"] = pd.array([pd.NA] * len(df), dtype="Int64")  # INT64

    # Alias for downstream (simulate_purchases, line_items, operations, aftermath)
    df["variant_id"] = df["id"].values

    return df


def simulate_purchases_from_clicks(
    clicks_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    products_df: pd.DataFrame,
    config: dict,
    creatives_df: pd.DataFrame | None = None,
    ads_df: Optional[pd.DataFrame] = None,
    prospects_df: Optional[pd.DataFrame] = None,
    brand_state=None,
    seasonality_mult: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert ad clicks into a behavioral funnel and potential purchases (orders, line items, transactions).

    Funnel is causal, not random rates:
        click → landing_page_view → add_to_cart → initiate_checkout → purchase.

    Tier 8 — Landing Page Experience: each click encounters a page experience derived from
    creative promise × product reality × human state. Five internal signals (coherence, clarity,
    cognitive_load, perceived_value, friction) modulate every funnel stage. These are internal
    simulation mechanics — nothing is written to warehouse. dbt infers all meaning.

    Landing page view depends on both psychological readiness (alignment, trust, impulse,
    fatigue, disappointment) AND experience encounter (coherence, clarity, cognitive load,
    friction, perceived value). Clicks that fail LPV become bounces.

    Add-to-cart is further influenced by page clarity and perceived value presentation.
    Hesitation delay increases with cognitive load and poor experience quality.
    Checkout initiation is modulated by the experience quality factor.

    When prospects_df is provided, clicks from prospects that convert create a new
    customer on first purchase (first-purchase-creates-customer).

    Mechanics:
    - Merge clicks with (prospects + customers) for traits; compute trait–creative alignment.
    - Compute landing page experience from creative promises × product attributes × human state.
    - Draw landing_page_view per click (psychology × experience); for LPV rows draw add_to_cart;
      for cart rows draw hesitation (delay) and checkout; for checkout rows draw purchase.
    - If a converting click is from a prospect: create new customer.
    - Returns (orders_df, line_items_df, transactions_df, new_customers_df, funnel_events_df).
    - funnel_events_df: event_type in ('landing_page_view','add_to_cart','initiate_checkout').

    Parameters
    ----------
    clicks_df : pandas.DataFrame
        One row per click. Must have: customer_id, date (or click_date).
    customers_df : pandas.DataFrame
        Must have: customer_id, impulse_level, loyalty_propensity,
        price_sensitivity.
    variants_df : pandas.DataFrame
        Must have: variant_id, product_id, price.
    products_df : pandas.DataFrame
        Must have: product_id, title (for product_title on line items).
    config : dict
        Optional: base_conversion or conversion_rate (float, e.g. 0.02–0.05).
        Optional: seed (int).
    creatives_df : pandas.DataFrame, optional
        If provided with creative_id, used for promise/trait alignment and fatigue only.
    ads_df : pandas.DataFrame, optional
        If provided, used for attribution hierarchy on orders.
    prospects_df : pandas.DataFrame, optional
        Anonymous humans (prospect_id = customer_id in this df). When a click from
        a prospect converts, a new customer is created and returned in new_customers_df.

    Returns
    -------
    tuple of (orders_df, line_items_df, transactions_df, new_customers_df, funnel_events_df, checkouts_df)
        new_customers_df: customers created from prospects on first purchase (may be empty).
        funnel_events_df: add_to_cart and initiate_checkout events (ad_id, date, event_type) for event-driven reporting.
        checkouts_df: shopify_checkouts source shape (one row per initiate_checkout; completed_at set when order exists).
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    base_conversion = config.get("base_conversion") or config.get("conversion_rate", 0.035)
    base_conversion = float(base_conversion)

    # Upgrade 1: Brand lifecycle phase effect on CVR
    if brand_state is not None:
        base_conversion *= brand_state.get_phase_effects().get("cvr_mult", 1.0)
        # Acquisition difficulty: harder to convert new prospects over time
        if brand_state.acquisition_difficulty_multiplier > 1.0:
            base_conversion /= brand_state.acquisition_difficulty_multiplier

    # Upgrade 4: Seasonality multiplier on conversion
    base_conversion *= seasonality_mult

    currency = config.get("currency") or "USD"

    # Resolve date column
    click_date_col = "date" if "date" in clicks_df.columns else "click_date"
    if click_date_col not in clicks_df.columns:
        raise ValueError("clicks_df must have 'date' or 'click_date'")

    _empty_order_columns = [
        "id", "order_number", "checkout_id", "cart_token",
        "created_at", "updated_at", "processed_at", "cancelled_at", "closed_at",
        "financial_status", "fulfillment_status", "order_status_url", "confirmed", "test",
        "currency", "subtotal_price", "total_price", "total_tax", "total_discounts",
        "total_shipping_price", "total_refunded",
        "customer_id", "email", "source_name", "landing_site", "referring_site",
        "tags", "note", "shipping_address_country", "shipping_address_region",
        "billing_address_country", "billing_address_region", "billing_address_city",
        "last_attributed_ad_id", "last_attributed_adset_id", "last_attributed_campaign_id", "last_attributed_creative_id",
    ]

    _empty_line_item_columns = [
        "id", "order_id", "product_id", "variant_id", "sku",
        "product_title", "variant_title", "name", "vendor",
        "quantity", "fulfillable_quantity", "fulfillment_status",
        "price", "total_discount", "discount_allocations", "taxable",
        "requires_shipping", "tax_lines", "created_at", "updated_at",
        "gift_card", "properties", "origin_location_id",
    ]

    _empty_transaction_columns = [
        "id", "order_id", "gateway", "kind", "status", "test", "currency", "amount",
        "authorization_code", "created_at", "processed_at", "updated_at",
        "source_name", "receipt", "error_code", "gateway_reference", "admin_graphql_api_id",
    ]

    _empty_customers_columns = list(customers_df.columns) if not customers_df.empty else ["customer_id"]
    if "first_touch_ad_id" not in _empty_customers_columns and customers_df.empty:
        _empty_customers_columns = _empty_customers_columns + [
            "first_touch_ad_id", "first_touch_adset_id", "first_touch_campaign_id",
            "first_touch_creative_id", "first_touch_date",
        ]

    _empty_funnel_columns = ["ad_id", "adset_id", "campaign_id", "creative_id", "customer_id", "date", "event_type"]
    _empty_checkout_columns = [
        "id", "token", "cart_token", "customer_id",
        "created_at", "updated_at", "completed_at", "closed_at",
        "currency", "total_price", "subtotal_price", "total_tax", "total_discounts", "total_line_items_price",
        "source_name", "landing_site", "referring_site", "abandoned_checkout_url", "email",
    ]
    if clicks_df.empty:
        empty_cust = pd.DataFrame(columns=_empty_customers_columns)
        return (
            pd.DataFrame(columns=_empty_order_columns),
            pd.DataFrame(columns=_empty_line_item_columns),
            pd.DataFrame(columns=_empty_transaction_columns),
            empty_cust,
            pd.DataFrame(columns=_empty_funnel_columns),
            pd.DataFrame(columns=_empty_checkout_columns),
            pd.DataFrame(),
        )

    # Attribution: enrich clicks with adset/campaign from ads so orders can carry attributed_*
    base_clicks = clicks_df.copy()
    if (
        ads_df is not None
        and not ads_df.empty
        and "ad_id" in clicks_df.columns
        and "id" in ads_df.columns
    ):
        ad_cols = ["id", "adset_id", "campaign_id"]
        ad_cols = [c for c in ad_cols if c in ads_df.columns]
        if ad_cols:
            ad_lookup = ads_df[ad_cols].rename(columns={"id": "ad_id"})
            base_clicks = clicks_df.merge(ad_lookup, on="ad_id", how="left")

    # Traits (and address) source: prospects + customers when prospects_df provided, else customers only
    trait_cols = ["customer_id", "impulse_level", "loyalty_propensity", "price_sensitivity"]
    if "trust_score" in customers_df.columns:
        trait_cols.append("trust_score")
    if "expressed_desire_level" in customers_df.columns:
        trait_cols.append("expressed_desire_level")
    for col in ["disappointment_memory", "recent_negative_velocity", "discount_dependency", "quality_expectation", "regret_propensity", "spending_propensity", "discount_only_buyer"]:
        if col in customers_df.columns and col not in trait_cols:
            trait_cols.append(col)
    addr_cols_for_merge = ["customer_id", "default_address_country", "default_address_region"]
    addr_cols_for_merge = [c for c in addr_cols_for_merge if c in customers_df.columns]
    merge_cols = list(dict.fromkeys(trait_cols + addr_cols_for_merge))

    if prospects_df is not None and not prospects_df.empty:
        pc_cols = [c for c in merge_cols if c in prospects_df.columns and c in customers_df.columns]
        prospects_traits = prospects_df[pc_cols].copy()
        customers_traits = customers_df[pc_cols].copy()
        combined_traits = pd.concat([prospects_traits, customers_traits], ignore_index=True)
        merged = base_clicks.merge(combined_traits, on="customer_id", how="left")
    else:
        merged = base_clicks.merge(
            customers_df[[c for c in merge_cols if c in customers_df.columns]],
            on="customer_id",
            how="left",
        )

    # Trait–creative alignment for cart/checkout/purchase (reuse psychological_state; prospects get 0.5 when missing)
    if creatives_df is not None and not creatives_df.empty and "creative_id" in merged.columns:
        merged["_alignment"] = merged.apply(
            lambda r: _trait_creative_alignment(r, r.get("creative_id"), creatives_df), axis=1
        )
    else:
        merged["_alignment"] = 0.5

    # Tier 4 creative fatigue: ensure merged has "fatigue" for funnel (atc/checkout) and CVR
    fatigue_beta_cvr = float(config.get("fatigue_beta_conversion", 0.08))
    if (
        "creative_fatigue_map" in customers_df.columns
        and "creative_id" in merged.columns
    ):
        fatigue_rows = []
        for _, c in customers_df.iterrows():
            m = c.get("creative_fatigue_map")
            if not isinstance(m, dict):
                continue
            for crid, f in m.items():
                if float(f) >= 0.01:
                    fatigue_rows.append({"customer_id": c["customer_id"], "creative_id": str(crid), "fatigue": float(f)})
        if fatigue_rows:
            fatigue_df = pd.DataFrame(fatigue_rows)
            merged = merged.merge(fatigue_df, on=["customer_id", "creative_id"], how="left")
            merged["fatigue"] = merged["fatigue"].fillna(0)
        else:
            merged["fatigue"] = np.zeros(len(merged))
    else:
        merged["fatigue"] = np.zeros(len(merged))

    # Price & discount psychology: draw per click (used in atc, checkout, and purchase)
    discount_prob = float(config.get("discount_probability", 0.30))
    discount_levels = config.get("discount_levels") or [0.1, 0.2, 0.3, 0.4]
    discount_levels = [float(x) for x in discount_levels]
    discount_boost = float(config.get("discount_conversion_boost", 0.5))
    has_discount = rng.random(size=len(merged)) < discount_prob
    _discount_pct = np.zeros(len(merged))
    _discount_code = [None] * len(merged)
    if len(discount_levels) > 0:
        for i in range(len(merged)):
            if has_discount[i]:
                pct = float(discount_levels[rng.integers(0, len(discount_levels))])
                _discount_pct[i] = pct
                _discount_code[i] = "PROMO_" + str(int(pct * 100))
    merged["_discount_pct"] = _discount_pct
    merged["_discount_code"] = _discount_code

    # --- Tier 8: Landing Page Experience (internal encounter, not warehouse) ---
    # Compute per-click experience signals from creative promises × product reality × human state.
    # These are internal simulation mechanics that modulate every funnel stage.
    # Nothing below is written to warehouse — dbt infers all meaning from behavioral outcomes.

    # Bring product attributes into the click context.
    n_merged = len(merged)
    _prod_quality_num = np.full(n_merged, 0.5)
    _prod_complexity = np.full(n_merged, 0.5)
    _prod_shipping_hard = np.zeros(n_merged)
    if products_df is not None and not products_df.empty:
        ql_map = {"low": 0.25, "mid": 0.55, "high": 0.85}
        sd_map = {"easy": 0.0, "normal": 0.3, "hard": 0.7}
        prod_idx = rng.integers(0, len(products_df), size=n_merged)
        _prod_quality_num = np.array([ql_map.get(products_df.iloc[i].get("quality_level", "mid"), 0.5) for i in prod_idx])
        _prod_complexity = np.array([float(products_df.iloc[i].get("complexity_level", 0.5)) for i in prod_idx])
        _prod_shipping_hard = np.array([sd_map.get(products_df.iloc[i].get("shipping_difficulty", "normal"), 0.3) for i in prod_idx])
    merged["_prod_quality"] = _prod_quality_num
    merged["_prod_complexity"] = _prod_complexity

    # Compute creative promise pressure per click (average intensity of 8-axis promises)
    _promise_pressure = np.full(n_merged, 0.4)
    if creatives_df is not None and not creatives_df.empty and "creative_id" in merged.columns:
        from .psychological_state import _promise_pressure_from_creative
        _promise_pressure = merged["creative_id"].map(
            lambda cid: _promise_pressure_from_creative(cid, creatives_df)
        ).fillna(0.4).values

    # --- Tier 9: merge creative strategy layers for experience modulation ---
    _angle_arr = merged.get("creative_angle")
    _hook_arr = merged.get("creative_hook_pattern")
    _persona_arr = merged.get("creative_persona")
    _vp_arr = merged.get("creative_value_proposition")
    _ct_arr = merged.get("creative_type")
    if creatives_df is not None and not creatives_df.empty and "creative_id" in merged.columns:
        tier9_cols = ["creative_id"]
        for c in ["creative_angle", "creative_hook_pattern", "creative_persona",
                   "creative_value_proposition", "creative_type"]:
            if c in creatives_df.columns and c not in merged.columns:
                tier9_cols.append(c)
        if len(tier9_cols) > 1:
            merged = merged.merge(
                creatives_df[tier9_cols].drop_duplicates("creative_id"),
                on="creative_id", how="left",
            )
        _angle_arr = merged.get("creative_angle")
        _hook_arr = merged.get("creative_hook_pattern")
        _persona_arr = merged.get("creative_persona")
        _vp_arr = merged.get("creative_value_proposition")
        _ct_arr = merged.get("creative_type")

    # Per-click angle expectation lift: some angles raise expectation baseline
    _angle_expectation_lift = np.zeros(n_merged)
    if _angle_arr is not None:
        for ang, params in _ANGLE_PARAMS.items():
            mask = (_angle_arr == ang).values
            _angle_expectation_lift[mask] = params["expectation_lift"]

    alignment = merged["_alignment"].values
    trust = merged["trust_score"].fillna(0.5).values if "trust_score" in merged.columns else np.full(n_merged, 0.5)
    disappointment = merged["disappointment_memory"].fillna(0).values if "disappointment_memory" in merged.columns else np.zeros(n_merged)
    recent_neg_vel = merged["recent_negative_velocity"].fillna(0).values if "recent_negative_velocity" in merged.columns else np.zeros(n_merged)
    discount_dep = merged["discount_dependency"].fillna(0).values if "discount_dependency" in merged.columns else np.zeros(n_merged)
    fatigue_arr = merged["fatigue"].values
    impulse = merged["impulse_level"].fillna(0.5).values
    price_sens = merged["price_sensitivity"].fillna(0.5).values
    quality_exp = merged["quality_expectation"].fillna(0.5).values if "quality_expectation" in merged.columns else np.full(n_merged, 0.5)

    # 1. Coherence: how well creative promise matches product reality.
    #    Tier 9: angle expectation lift widens the gap between promise and reality
    #    when product quality is low — making coherence worse for high-expectation angles.
    effective_pressure = _promise_pressure + _angle_expectation_lift
    _coherence = np.clip(
        1.0 - np.abs(effective_pressure - _prod_quality_num)
        + 0.15 * alignment
        + rng.normal(0, 0.03, size=n_merged),
        0.0, 1.0,
    )

    # 2. Clarity
    _clarity = np.clip(
        1.0 - 0.5 * _prod_complexity
        + 0.2 * alignment
        - 0.1 * (1.0 - trust)
        + rng.normal(0, 0.02, size=n_merged),
        0.0, 1.0,
    )

    # 3. Cognitive load: angle expectation lift adds mental effort (high promise = more to process)
    _cognitive_load = np.clip(
        0.3 * _prod_complexity
        + 0.15 * fatigue_arr
        + 0.2 * np.abs(effective_pressure - _prod_quality_num)
        + 0.1 * (1.0 - alignment)
        + 0.05 * _angle_expectation_lift
        + rng.normal(0, 0.02, size=n_merged),
        0.0, 1.0,
    )

    # 4. Perceived value
    _perceived_value = np.clip(
        0.3 * _prod_quality_num
        + 0.2 * alignment
        + 0.15 * merged["_discount_pct"].values
        - 0.25 * price_sens * np.maximum(0, effective_pressure - _prod_quality_num)
        + rng.normal(0, 0.02, size=n_merged),
        0.0, 1.0,
    )

    # 5. Friction
    _friction = np.clip(
        0.2 * _prod_shipping_hard
        + 0.1 * _prod_complexity
        + 0.08 * (1.0 - trust)
        + rng.normal(0, 0.02, size=n_merged),
        0.0, 1.0,
    )

    _experience_quality = np.clip(
        0.30 * _coherence
        + 0.25 * _clarity
        + 0.20 * _perceived_value
        + 0.15 * (1.0 - _cognitive_load)
        + 0.10 * (1.0 - _friction),
        0.0, 1.0,
    )
    merged["_experience_quality"] = _experience_quality

    # --- Behavioral funnel: click → landing_page_view → add_to_cart → initiate_checkout → purchase ---

    # 0) Landing page view: psychological readiness × experience encounter.
    lpv_psych_boost = (
        0.08 * alignment
        + 0.03 * impulse
        + 0.06 * trust
    )
    lpv_psych_penalty = (
        0.05 * fatigue_arr
        + 0.06 * disappointment
        + 0.03 * recent_neg_vel
        + 0.04 * price_sens * np.where(
            (merged["_discount_pct"].values <= 0) & (discount_dep > 0.3),
            1.0, 0.0,
        )
    )
    lpv_experience_boost = (
        0.08 * _coherence
        + 0.06 * _clarity
        + 0.04 * _perceived_value
    )
    lpv_experience_penalty = (
        0.06 * _cognitive_load
        + 0.05 * _friction
        + 0.04 * quality_exp * (1.0 - _coherence)
    )

    # Tier 9: hook bounce risk — some hooks create attention then lose it on the page
    _hook_bounce_risk = np.zeros(n_merged)
    if _hook_arr is not None:
        for hook, params in _HOOK_PARAMS.items():
            mask = (_hook_arr == hook).values
            _hook_bounce_risk[mask] = params["bounce_risk"]

    # Tier 9: persona resonance reduces bounce (aligned human stays longer)
    _funnel_persona_resonance = np.full(n_merged, 0.5)
    if _persona_arr is not None:
        for i in range(n_merged):
            persona = _persona_arr.iat[i] if hasattr(_persona_arr, "iat") else None
            if persona is not None and pd.notna(persona):
                trait_vals = {
                    "price_sensitivity": float(price_sens[i]),
                    "impulse_level": float(impulse[i]),
                    "quality_expectation": float(quality_exp[i]),
                    "regret_propensity": float(merged["regret_propensity"].fillna(0.5).iat[i]) if "regret_propensity" in merged.columns else 0.5,
                    "loyalty_propensity": float(merged["loyalty_propensity"].fillna(0.5).iat[i]) if "loyalty_propensity" in merged.columns else 0.5,
                }
                _funnel_persona_resonance[i] = _persona_resonance(persona, trait_vals)

    lpv_noise = rng.normal(0, 0.01, size=n_merged)

    base_lpv = float(config.get("base_landing_page_view", 0.72))
    lpv_prob = (
        base_lpv
        + lpv_psych_boost - lpv_psych_penalty
        + lpv_experience_boost - lpv_experience_penalty
        - _hook_bounce_risk
        + 0.02 * (_funnel_persona_resonance - 0.5)
        + lpv_noise
    )
    lpv_prob = np.clip(lpv_prob, 0.0, 1.0)
    lpv_passed = rng.random(size=n_merged) < lpv_prob

    merged_lpv = merged.loc[lpv_passed].copy()

    # Carry experience signals into LPV-filtered subset for downstream funnel stages
    exp_quality_lpv = merged_lpv["_experience_quality"].values
    clarity_lpv = _clarity[lpv_passed]
    coherence_lpv = _coherence[lpv_passed]
    cognitive_load_lpv = _cognitive_load[lpv_passed]
    perceived_value_lpv = _perceived_value[lpv_passed]
    friction_lpv = _friction[lpv_passed]
    trust_lpv = merged_lpv["trust_score"].fillna(0.5).values if "trust_score" in merged_lpv.columns else np.full(len(merged_lpv), 0.5)
    disappointment_lpv = merged_lpv["disappointment_memory"].fillna(0).values if "disappointment_memory" in merged_lpv.columns else np.zeros(len(merged_lpv))
    recent_neg_vel_lpv = merged_lpv["recent_negative_velocity"].fillna(0).values if "recent_negative_velocity" in merged_lpv.columns else np.zeros(len(merged_lpv))
    alignment_lpv = merged_lpv["_alignment"].values
    fatigue_lpv = merged_lpv["fatigue"].values

    # 1) Add-to-cart: human psychology + experience clarity and perceived value
    base_atc = float(config.get("base_add_to_cart", 0.42))
    impulse_lift_atc = 0.04 * merged_lpv["impulse_level"].values
    trust_lift_atc = 0.03 * trust_lpv
    price_penalty_atc = 0.03 * merged_lpv["price_sensitivity"].values
    disappointment_penalty = 0.06 * disappointment_lpv
    neg_vel_penalty = 0.05 * recent_neg_vel_lpv
    alignment_lift = 0.05 * alignment_lpv
    discount_boost_atc = 0.04 * merged_lpv["_discount_pct"].values
    fatigue_penalty_atc = 0.03 * fatigue_lpv
    experience_lift_atc = 0.05 * clarity_lpv + 0.04 * perceived_value_lpv
    experience_penalty_atc = 0.04 * cognitive_load_lpv
    noise_atc = rng.normal(0, 0.008, size=len(merged_lpv))
    atc_prob = (
        base_atc + impulse_lift_atc + trust_lift_atc + alignment_lift + discount_boost_atc
        + experience_lift_atc
        - price_penalty_atc - disappointment_penalty - neg_vel_penalty
        - fatigue_penalty_atc - experience_penalty_atc
        + noise_atc
    )
    atc_prob = np.clip(atc_prob, 0.0, 1.0)
    cart_added_lpv = rng.random(size=len(merged_lpv)) < atc_prob
    merged_cart = merged_lpv.loc[cart_added_lpv].copy()

    # 2) Hesitation (delay): cognitive friction from human state + experience cognitive load
    if len(merged_cart) > 0:
        regret = merged_cart["regret_propensity"].fillna(0.5).values if "regret_propensity" in merged_cart.columns else np.full(len(merged_cart), 0.5)
        trust_cart = merged_cart["trust_score"].fillna(0.5).values if "trust_score" in merged_cart.columns else np.full(len(merged_cart), 0.5)
        align_cart = merged_cart["_alignment"].values
        cog_load_cart = merged_cart["_experience_quality"].values
        base_delay = (
            5.0
            + 40.0 * regret
            + 25.0 * (1.0 - trust_cart)
            + 15.0 * (1.0 - align_cart)
            + 12.0 * (1.0 - cog_load_cart)
        )
        delay_noise = rng.normal(0, 8.0, size=len(merged_cart))
        merged_cart["_delay_minutes"] = np.clip(
            (base_delay + delay_noise).astype(int), 1, 120
        )
    else:
        merged_cart["_delay_minutes"] = pd.Series(dtype=int)

    # 3) Initiate checkout: trust × alignment × experience, with hesitation and friction penalties
    if len(merged_cart) > 0:
        base_checkout = float(config.get("base_checkout", 0.72))
        checkout_trust = merged_cart["trust_score"].fillna(0.5).values if "trust_score" in merged_cart.columns else np.full(len(merged_cart), 0.5)
        checkout_align = merged_cart["_alignment"].values
        checkout_discount_boost = 0.2 * merged_cart["_discount_pct"].values
        dep = merged_cart["discount_dependency"].fillna(0).values if "discount_dependency" in merged_cart.columns else np.zeros(len(merged_cart))
        no_discount = (merged_cart["_discount_pct"].fillna(0).values <= 0.0)
        dependency_penalty = np.where(no_discount & (dep > 0.3), 0.65, 1.0)
        hesitation_penalty = np.exp(-merged_cart["_delay_minutes"].values.astype(float) / 35.0)
        fatigue_penalty_co = np.exp(-fatigue_beta_cvr * merged_cart["fatigue"].values)
        exp_q_cart = merged_cart["_experience_quality"].values
        experience_checkout_factor = 0.85 + 0.15 * exp_q_cart
        checkout_prob = (
            base_checkout * checkout_trust * (1.0 + checkout_discount_boost) * checkout_align
            * hesitation_penalty * fatigue_penalty_co * dependency_penalty
            * experience_checkout_factor
        )
        checkout_prob = np.clip(checkout_prob, 0.0, 1.0)
        checkout_initiated = rng.random(size=len(merged_cart)) < checkout_prob
    else:
        checkout_initiated = np.array([], dtype=bool)
    merged_checkout = merged_cart.loc[checkout_initiated].copy() if len(merged_cart) > 0 else pd.DataFrame()

    # 4) Purchase: existing CVR logic applied only to checkout rows (not to all clicks)
    if len(merged_checkout) > 0:
        impulse_lift = 0.03 * merged_checkout["impulse_level"].values
        loyalty_lift = 0.03 * merged_checkout["loyalty_propensity"].values
        price_penalty = 0.04 * merged_checkout["price_sensitivity"].values
        noise = rng.normal(0, 0.005, size=len(merged_checkout))
        conv_prob = base_conversion + impulse_lift + loyalty_lift - price_penalty + noise
        conv_prob = np.clip(conv_prob, 0.0, 1.0)
        trust_cv = merged_checkout["trust_score"].fillna(0.5).values if "trust_score" in merged_checkout.columns else np.full(len(merged_checkout), 0.5)
        conv_prob = conv_prob * trust_cv
        conv_prob = np.clip(conv_prob, 0.0, 1.0)
        conv_prob = conv_prob * np.exp(-merged_checkout["fatigue"].values * fatigue_beta_cvr)
        conv_prob = np.clip(conv_prob, 0.0, 1.0)
        conv_prob = conv_prob * (1.0 + merged_checkout["_discount_pct"].values * discount_boost)
        conv_prob = np.clip(conv_prob, 0.0, 1.0)
        # Upgrade 6: discount-only buyers convert poorly at full price
        if "discount_only_buyer" in merged_checkout.columns:
            disc_only = merged_checkout["discount_only_buyer"].fillna(False).values.astype(bool)
            no_discount = merged_checkout["_discount_pct"].values < 0.01
            disc_only_penalty = float(config.get("memory", {}).get("discount_only_fullprice_penalty", 0.4))
            conv_prob = np.where(disc_only & no_discount, conv_prob * (1.0 - disc_only_penalty), conv_prob)
            conv_prob = np.clip(conv_prob, 0.0, 1.0)
        desire_weight = float(config.get("desire_conversion_weight", 0.4))
        if "expressed_desire_level" in merged_checkout.columns:
            desire = merged_checkout["expressed_desire_level"].fillna(0.1).values
            conv_prob = conv_prob * (1.0 + desire * desire_weight)
        conv_prob = np.clip(conv_prob, 0.0, 1.0)
        # Time-based conversion ramp: brand grows over time, so later years get higher CVR
        sim = config.get("simulation") or {}
        sim_start = pd.Timestamp(sim.get("start_date", "2023-01-01")).normalize()
        click_date = pd.Timestamp(merged_checkout[click_date_col].iloc[0]).normalize()
        years_since_start = (click_date - sim_start).days / 365.0
        ramp_year_0 = float(config.get("conversion_ramp_year_0", 0.75))
        ramp_year_2_plus = float(config.get("conversion_ramp_year_2_plus", 1.25))
        time_multiplier = ramp_year_0 + (ramp_year_2_plus - ramp_year_0) * min(years_since_start / 2.0, 1.0)
        conv_prob = conv_prob * time_multiplier
        conv_prob = np.clip(conv_prob, 0.0, 1.0)
        purchased = rng.random(size=len(merged_checkout)) < conv_prob
        merged_purchased = merged_checkout.loc[purchased].copy()
    else:
        merged_purchased = pd.DataFrame()

    # Tier 6: zero-purchase days are realistic; no forced conversions.
    # Persistent cart memory and delayed returns produce conversions over multi-day windows.

    # Abandoned carts: humans who added to cart but did NOT purchase today (for persistent cart memory)
    _abandoned_cart_cols = ["customer_id", "ad_id", "adset_id", "campaign_id", "creative_id", "_discount_pct", "_discount_code", "_alignment", click_date_col]
    _abandoned_cart_cols = [c for c in _abandoned_cart_cols if c in merged.columns]
    if len(merged_cart) > 0:
        purchased_ids = set(merged_purchased["customer_id"].astype(str)) if not merged_purchased.empty else set()
        abandoned_mask = ~merged_cart["customer_id"].astype(str).isin(purchased_ids)
        abandoned_carts_df = merged_cart.loc[abandoned_mask, _abandoned_cart_cols].copy() if abandoned_mask.any() else pd.DataFrame()
        if not abandoned_carts_df.empty:
            abandoned_carts_df = abandoned_carts_df.drop_duplicates(subset=["customer_id"], keep="last")
            abandoned_carts_df["date"] = pd.to_datetime(abandoned_carts_df[click_date_col]).dt.strftime("%Y-%m-%d")
    else:
        abandoned_carts_df = pd.DataFrame()

    # Funnel events for Meta reporting (event-driven landing_page_view / add_to_cart / initiate_checkout)
    funnel_cols = ["ad_id", "adset_id", "campaign_id", "creative_id", "customer_id", click_date_col]
    funnel_cols = [c for c in funnel_cols if c in merged.columns]
    lpv_rows = merged.loc[lpv_passed, funnel_cols].copy() if lpv_passed.any() else pd.DataFrame()
    if not lpv_rows.empty:
        lpv_rows["event_type"] = "landing_page_view"
        lpv_rows["date"] = pd.to_datetime(lpv_rows[click_date_col]).dt.strftime("%Y-%m-%d")
    atc_rows = merged_lpv.loc[cart_added_lpv, funnel_cols].copy() if cart_added_lpv.any() else pd.DataFrame()
    if not atc_rows.empty:
        atc_rows["event_type"] = "add_to_cart"
        atc_rows["date"] = pd.to_datetime(atc_rows[click_date_col]).dt.strftime("%Y-%m-%d")
    co_rows = merged_cart.loc[checkout_initiated, funnel_cols].copy() if len(merged_cart) > 0 and checkout_initiated.any() else pd.DataFrame()
    if not co_rows.empty:
        co_rows["event_type"] = "initiate_checkout"
        co_rows["date"] = pd.to_datetime(co_rows[click_date_col]).dt.strftime("%Y-%m-%d")
    funnel_parts = [df for df in [lpv_rows, atc_rows, co_rows] if not df.empty]
    funnel_events_df = pd.concat(funnel_parts, ignore_index=True) if funnel_parts else pd.DataFrame(columns=_empty_funnel_columns)

    # Build shopify_checkouts (one row per initiate_checkout; completed_at set when order exists)
    def _build_checkouts_from_funnel(merged_checkout_df: pd.DataFrame, checkout_idx_to_order: Optional[dict] = None) -> pd.DataFrame:
        if merged_checkout_df is None or merged_checkout_df.empty:
            return pd.DataFrame(columns=_empty_checkout_columns)
        checkout_idx_to_order = checkout_idx_to_order or {}
        rows = []
        for idx, row in merged_checkout_df.iterrows():
            checkout_id = str(uuid.uuid4())
            token = str(uuid.uuid4())
            cart_token = str(uuid.uuid4())
            created_at = pd.to_datetime(row[click_date_col]) + pd.Timedelta(minutes=int(row["_delay_minutes"]))
            order_row = checkout_idx_to_order.get(idx)
            if order_row is not None:
                completed_at = order_row["created_at"]
                closed_at = order_row["created_at"]
                total_price = float(order_row["total_price"])
                subtotal_price = float(order_row["subtotal_price"])
                total_tax = float(order_row["total_tax"])
                total_discounts = float(order_row["total_discounts"])
                total_line_items_price = float(order_row["subtotal_price"])
                abandoned_checkout_url = None
            else:
                completed_at = pd.NaT
                closed_at = pd.NaT
                total_price = 0.0
                subtotal_price = 0.0
                total_tax = 0.0
                total_discounts = 0.0
                total_line_items_price = 0.0
                abandoned_checkout_url = "https://store/checkout/recovery"
            rows.append({
                "id": checkout_id,
                "token": token,
                "cart_token": cart_token,
                "customer_id": row["customer_id"],
                "created_at": created_at,
                "updated_at": completed_at if order_row is not None else created_at,
                "completed_at": completed_at,
                "closed_at": closed_at,
                "currency": currency,
                "total_price": total_price,
                "subtotal_price": subtotal_price,
                "total_tax": total_tax,
                "total_discounts": total_discounts,
                "total_line_items_price": total_line_items_price,
                "source_name": "meta",
                "landing_site": None,
                "referring_site": None,
                "abandoned_checkout_url": abandoned_checkout_url,
                "email": None,
            })
        return pd.DataFrame(rows)

    checkouts_df = _build_checkouts_from_funnel(merged_checkout, None)

    if merged_purchased.empty:
        empty_cust = pd.DataFrame(columns=_empty_customers_columns)
        return (
            pd.DataFrame(columns=_empty_order_columns),
            pd.DataFrame(columns=_empty_line_item_columns),
            pd.DataFrame(columns=_empty_transaction_columns),
            empty_cust,
            funnel_events_df,
            checkouts_df,
            abandoned_carts_df,
        )

    # Mark prospect rows (first purchase will create customer)
    if prospects_df is not None and not prospects_df.empty:
        merged_purchased["_is_prospect"] = ~merged_purchased["customer_id"].astype(str).isin(
            customers_df["customer_id"].astype(str)
        )
    else:
        merged_purchased["_is_prospect"] = False

    # Merge customer default address only when we didn't already get it from combined_traits
    if prospects_df is None or prospects_df.empty:
        addr_cols = ["customer_id", "default_address_country", "default_address_region"]
        addr_cols = [c for c in addr_cols if c in customers_df.columns]
        if addr_cols:
            merged_purchased = merged_purchased.merge(
                customers_df[addr_cols].drop_duplicates("customer_id"),
                on="customer_id",
                how="left",
            )

    # Chronological first-touch: use hesitation delay from funnel (already on merged_purchased) or fallback draw
    merged_purchased = merged_purchased.copy()
    if "_delay_minutes" not in merged_purchased.columns or merged_purchased["_delay_minutes"].isna().any():
        merged_purchased["_delay_minutes"] = rng.integers(1, 31, size=len(merged_purchased))
    merged_purchased["_order_ts"] = (
        pd.to_datetime(merged_purchased[click_date_col]) + pd.to_timedelta(merged_purchased["_delay_minutes"], unit="m")
    )
    sort_by = ["_order_ts"]
    for col in [click_date_col, "ad_id", "creative_id"]:
        if col in merged_purchased.columns:
            sort_by.append(col)
    merged_purchased = merged_purchased.sort_values(
        by=sort_by, ascending=[True] * len(sort_by), kind="mergesort", na_position="last"
    )

    n_purchases = len(merged_purchased)
    variant_list = variants_df.to_dict("records")
    n_variants = len(variant_list)
    if n_variants == 0:
        raise ValueError("variants_df is empty")

    orders_rows: List[dict] = []
    line_items_rows: List[dict] = []
    transactions_rows: List[dict] = []
    new_customers_rows: List[dict] = []
    prospect_to_customer: dict = {}  # chronological first conversion per prospect creates customer; later reuse, never overwrite

    for i, (_, row) in enumerate(merged_purchased.iterrows()):
        order_id = str(uuid.uuid4())
        prospect_id = row["customer_id"]
        is_prospect = row.get("_is_prospect", False)
        order_timestamp = pd.Timestamp(row["_order_ts"])
        minutes_delay = int(row["_delay_minutes"])

        if is_prospect and prospects_df is not None and not prospects_df.empty:
            prospect_key = str(prospect_id)
            if prospect_key in prospect_to_customer:
                customer_id = prospect_to_customer[prospect_key]
                # Later conversion: reuse customer_id only; never overwrite signup_date or first_touch_*
            else:
                customer_id = "cust_" + str(prospect_id)
                prospect_to_customer[prospect_key] = customer_id
                prows = prospects_df[prospects_df["customer_id"].astype(str) == prospect_key]
                if not prows.empty:
                    prow = prows.iloc[0]
                    signup_date = pd.Timestamp(order_timestamp).normalize()
                    nc = {
                    "id": customer_id,
                    "customer_id": customer_id,
                    "created_at": signup_date,
                    "updated_at": signup_date,
                    "signup_date": signup_date,
                    "price_sensitivity": prow.get("price_sensitivity", 0.5),
                    "impulse_level": prow.get("impulse_level", 0.5),
                    "loyalty_propensity": prow.get("loyalty_propensity", 0.5),
                    "regret_propensity": prow.get("regret_propensity", 0.5),
                    "quality_expectation": prow.get("quality_expectation", 0.5),
                    "income_proxy": prow.get("income_proxy", "mid"),
                    "acquisition_channel_preference": prow.get("acquisition_channel_preference", "meta"),
                    # Upgrade 5: acquisition pressure penalizes new customer trust
                    "trust_score": max(0.1, 0.5 - (brand_state.acquisition_pressure * float(config.get("feedback_loops", {}).get("pressure_quality_penalty", 0.1)) if brand_state is not None else 0)),
                    "disappointment_memory": 0.0,
                    "satisfaction_memory": 0.0,
                    "recent_negative_velocity": 0.0,
                    "discount_dependency": 0.0,
                    "exposure_count": 0,
                    "expressed_desire_level": 0.1,
                    "desire_decay_memory": 0.0,
                    "creative_fatigue_map": {},
                    "cart_memory": None,
                    "last_attributed_ad_id": row.get("ad_id"),
                    "last_attributed_adset_id": row.get("adset_id"),
                    "last_attributed_campaign_id": row.get("campaign_id"),
                    "last_attributed_creative_id": row.get("creative_id"),
                    "last_exposure_date": pd.NaT,
                    "last_order_date": pd.NaT,
                    "last_order_at": pd.NaT,
                    "first_touch_ad_id": row.get("ad_id"),
                    "first_touch_adset_id": row.get("adset_id"),
                    "first_touch_campaign_id": row.get("campaign_id"),
                    "first_touch_creative_id": row.get("creative_id"),
                    "first_touch_date": signup_date,
                    "state": prow.get("state", "enabled"),
                    "verified_email": prow.get("verified_email", True),
                    "email": prow.get("email"),
                    "first_name": prow.get("first_name"),
                    "last_name": prow.get("last_name"),
                    "accepts_marketing": prow.get("accepts_marketing", False),
                    "default_address_country": prow.get("default_address_country"),
                    "default_address_region": prow.get("default_address_region"),
                    "default_address_city": prow.get("default_address_city"),
                }
                for col in ["phone", "locale", "tags", "note", "source_name", "display_name", "default_address_id", "default_address_postal_code"]:
                    nc[col] = prow.get(col)
                nc["tax_exempt"] = pd.NA
                new_customers_rows.append(nc)
        else:
            customer_id = prospect_id

        order_row = {
            "order_id": order_id,
            "customer_id": customer_id,
            "order_timestamp": order_timestamp,
            "hesitation_proxy": minutes_delay,
        }
        # Attribution: the click that produced this order carries the ad hierarchy
        if "ad_id" in row.index:
            order_row["last_attributed_ad_id"] = row.get("ad_id")
        else:
            order_row["last_attributed_ad_id"] = None
        order_row["last_attributed_adset_id"] = row.get("adset_id") if "adset_id" in row.index else None
        order_row["last_attributed_campaign_id"] = row.get("campaign_id") if "campaign_id" in row.index else None
        order_row["last_attributed_creative_id"] = row.get("creative_id") if "creative_id" in row.index else None
        order_row["discount_pct"] = row.get("_discount_pct", 0.0)
        order_row["discount_code"] = row.get("_discount_code")
        # Order address enrichment: customer default as anchor (90% same, 7% diff region, 3% diff country)
        ship_country, ship_region, bill_country, bill_region = _enrich_order_addresses(
            row.get("default_address_country"),
            row.get("default_address_region"),
            rng,
        )
        order_row["shipping_address_country"] = ship_country
        order_row["shipping_address_region"] = ship_region
        order_row["billing_address_country"] = bill_country
        order_row["billing_address_region"] = bill_region
        order_row["billing_address_city"] = _pick_city_for_country(bill_country, rng)
        orders_rows.append(order_row)

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

    orders_df = pd.DataFrame(orders_rows)
    line_items_df = pd.DataFrame(line_items_rows)
    transactions_df = pd.DataFrame(transactions_rows)
    new_customers_df = pd.DataFrame(new_customers_rows) if new_customers_rows else pd.DataFrame(columns=_empty_customers_columns)

    # Enrich transactions_df to Shopify shopify_transactions source schema (typed nulls per contract)
    transactions_df = transactions_df.rename(columns={"transaction_id": "id"})
    for col in ["gateway", "kind", "authorization_code", "source_name", "receipt", "error_code", "gateway_reference", "admin_graphql_api_id"]:
        transactions_df[col] = None  # STRING (object)
    transactions_df["test"] = pd.array([pd.NA] * len(transactions_df), dtype="boolean")  # BOOL
    transactions_df["processed_at"] = pd.NaT  # TIMESTAMP
    transactions_df["updated_at"] = pd.NaT  # TIMESTAMP
    # currency, amount, created_at from order (set after orders_df is built below)

    # Upgrade 7: spending_propensity multiplies line item price (whales buy premium)
    if not orders_df.empty and "spending_propensity" in merged_purchased.columns:
        order_to_spend = dict(zip(
            orders_df["order_id"] if "order_id" in orders_df.columns else orders_df["id"],
            merged_purchased["spending_propensity"].fillna(1.0).values
        ))
        line_items_df = line_items_df.copy()
        line_items_df["price"] = line_items_df.apply(
            lambda r: round(float(r["price"]) * float(order_to_spend.get(r["order_id"], 1.0)), 2),
            axis=1,
        )
    # Subtotal per order (sum of line item price * quantity)
    line_items_df = line_items_df.copy()
    line_items_df["_line_total"] = line_items_df["price"] * line_items_df["quantity"]
    subtotal_by_order = line_items_df.groupby("order_id")["_line_total"].sum()
    line_items_df = line_items_df.drop(columns=["_line_total"])

    # Enrich line_items_df to Shopify shopify_order_line_items source schema
    # Get product_id and variant title from variants_df
    variant_cols = ["variant_id", "product_id"]
    if "title" in variants_df.columns:
        variant_cols.append("title")
    variants_lookup = variants_df[variant_cols].drop_duplicates("variant_id")
    if "title" in variants_lookup.columns:
        variants_lookup = variants_lookup.rename(columns={"title": "variant_title"})
    line_items_df = line_items_df.merge(variants_lookup, on="variant_id", how="left")
    if "variant_title" not in line_items_df.columns:
        line_items_df["variant_title"] = None

    # Get product title from products_df
    if "product_id" in products_df.columns and "title" in products_df.columns:
        product_lookup = products_df[["product_id", "title"]].drop_duplicates("product_id").rename(
            columns={"title": "product_title"}
        )
        line_items_df = line_items_df.merge(product_lookup, on="product_id", how="left")
    if "product_title" not in line_items_df.columns:
        line_items_df["product_title"] = None

    line_items_df = line_items_df.merge(
        orders_df[["order_id", "order_timestamp"]],
        on="order_id",
        how="left",
    )
    line_items_df["created_at"] = line_items_df["order_timestamp"]
    line_items_df["updated_at"] = line_items_df["order_timestamp"]
    line_items_df = line_items_df.drop(columns=["order_timestamp"])
    line_items_df["fulfillable_quantity"] = line_items_df["quantity"]
    line_items_df["fulfillment_status"] = "fulfilled"
    line_items_df["taxable"] = True
    line_items_df["requires_shipping"] = True
    line_items_df["gift_card"] = False
    for col in ["sku", "vendor", "name", "discount_allocations", "tax_lines", "properties", "origin_location_id"]:
        line_items_df[col] = None  # STRING (object)
    line_items_df["price_set_presentment_amount"] = np.nan  # NUMERIC
    line_items_df["price_set_shop_amount"] = np.nan  # NUMERIC
    line_items_df["total_discount"] = np.nan  # NUMERIC
    line_items_df = line_items_df.rename(columns={"line_item_id": "id"})
    line_items_df = line_items_df[
        [
            "id", "order_id", "product_id", "variant_id", "sku",
            "product_title", "variant_title", "vendor", "name",
            "quantity", "fulfillable_quantity", "fulfillment_status",
            "price", "price_set_presentment_amount", "price_set_shop_amount",
            "total_discount", "discount_allocations", "taxable",
            "requires_shipping", "tax_lines",
            "created_at", "updated_at",
            "gift_card", "properties", "origin_location_id",
        ]
    ]

    orders_df["subtotal_price"] = orders_df["order_id"].map(subtotal_by_order).astype(float)
    orders_df["discount_pct"] = orders_df.get("discount_pct", 0.0).fillna(0)
    orders_df["total_discounts"] = (orders_df["subtotal_price"] * orders_df["discount_pct"]).round(2)
    orders_df["total_tax"] = ((orders_df["subtotal_price"] - orders_df["total_discounts"]) * 0.08).round(2)
    orders_df["total_shipping_price"] = 5.0
    orders_df["total_price"] = (
        orders_df["subtotal_price"]
        - orders_df["total_discounts"]
        + orders_df["total_tax"]
        + orders_df["total_shipping_price"]
    ).round(2)
    orders_df["total_refunded"] = 0.0
    if "discount_code" not in orders_df.columns:
        orders_df["discount_code"] = None

    orders_df["currency"] = currency
    orders_df["financial_status"] = "paid"
    orders_df["fulfillment_status"] = "fulfilled"
    orders_df["confirmed"] = True
    orders_df["test"] = False

    # Rename to Shopify source schema: order_id → id, order_timestamp → created_at
    orders_df = orders_df.rename(columns={"order_id": "id", "order_timestamp": "created_at"})
    orders_df["updated_at"] = orders_df["created_at"]
    orders_df["processed_at"] = orders_df["created_at"]

    # Transactions: currency, amount, created_at from order (order_id → orders.id)
    order_for_tx = orders_df[["id", "currency", "total_price", "created_at"]].rename(
        columns={"id": "order_id", "total_price": "amount"}
    )
    transactions_df = transactions_df.merge(order_for_tx, on="order_id", how="left")
    transactions_df = transactions_df[
        [
            "id", "order_id", "gateway", "kind", "status", "test", "currency", "amount",
            "authorization_code", "created_at", "processed_at", "updated_at",
            "source_name", "receipt", "error_code", "gateway_reference", "admin_graphql_api_id",
        ]
    ]

    # Email: join from customers (+ new customers from prospects) if column exists, else NULL
    customers_for_email = customers_df
    if not new_customers_df.empty and "email" in new_customers_df.columns:
        customers_for_email = pd.concat([customers_df, new_customers_df], ignore_index=True)
    if "email" in customers_for_email.columns:
        customer_email = customers_for_email[["customer_id", "email"]].drop_duplicates("customer_id")
        orders_df = orders_df.merge(customer_email, on="customer_id", how="left")
    else:
        orders_df["email"] = None

    # Rare / optional columns (NULL) — typed per contract: TIMESTAMP → pd.NaT, STRING → None
    for col in ["cancelled_at", "closed_at"]:
        if col not in orders_df.columns:
            orders_df[col] = pd.NaT  # TIMESTAMP
    for col in [
        "order_number", "checkout_id", "cart_token", "order_status_url", "source_name",
        "landing_site", "referring_site", "tags", "note",
        "shipping_address_country", "shipping_address_region",
        "billing_address_country", "billing_address_region", "billing_address_city",
    ]:
        if col not in orders_df.columns:
            orders_df[col] = None  # STRING (object)

    # Column order for Shopify-style raw table (attribution + discount for behavioral lineage)
    out_cols = [
        "id", "order_number", "checkout_id", "cart_token",
        "created_at", "updated_at", "processed_at", "cancelled_at", "closed_at",
        "financial_status", "fulfillment_status", "order_status_url", "confirmed", "test",
        "currency", "subtotal_price", "total_price", "total_tax", "total_discounts",
        "total_shipping_price", "total_refunded",
        "customer_id", "email", "source_name", "landing_site", "referring_site",
        "tags", "note", "shipping_address_country", "shipping_address_region",
        "billing_address_country", "billing_address_region", "billing_address_city",
        "last_attributed_ad_id", "last_attributed_adset_id", "last_attributed_campaign_id", "last_attributed_creative_id",
    ]
    for c in ["discount_pct", "discount_code", "hesitation_proxy"]:
        if c in orders_df.columns and c not in out_cols:
            out_cols.append(c)
    orders_df = orders_df[[c for c in out_cols if c in orders_df.columns]]

    # Keep aliases for downstream (simulate_refunds, simulate_fulfillments) without changing their code
    orders_df["order_id"] = orders_df["id"].values
    orders_df["order_timestamp"] = orders_df["created_at"].values

    # Rebuild checkouts with completed_at / totals from orders (checkout index -> order row)
    checkout_idx_to_order = {}
    if not merged_purchased.empty and not orders_df.empty:
        for i in range(len(merged_purchased)):
            idx = merged_purchased.index[i]
            order_row = orders_df.iloc[i]
            checkout_idx_to_order[idx] = order_row
    checkouts_df = _build_checkouts_from_funnel(merged_checkout, checkout_idx_to_order)

    return (orders_df, line_items_df, transactions_df, new_customers_df, funnel_events_df, checkouts_df, abandoned_carts_df)


def simulate_repeat_purchases_for_day(
    customers_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    products_df: pd.DataFrame,
    current_date: pd.Timestamp,
    config: dict,
    brand_state=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tier 3: additional orders from customers repeating without ad touch.
    Probability depends on trust_score, satisfaction_memory, disappointment_memory, recency.
    Attribution = last order's last_attributed_* for that customer (nullable if never had).
    Returns same schema as simulate_purchases_from_clicks; concat with ad-driven orders in main.
    """
    seed = config.get("seed")
    # Deterministic per day for reproducibility
    base = int(seed) if seed is not None else 0
    day_ordinal = pd.Timestamp(current_date).normalize().toordinal()
    rng = np.random.default_rng((base + day_ordinal) % (2**31))

    base_repeat = float(config.get("base_repeat_rate", 0.01))

    # Upgrade 1: Brand lifecycle phase effect on repeat rate
    if brand_state is not None:
        base_repeat *= brand_state.get_phase_effects().get("repeat_mult", 1.0)

    half_life = float(config.get("repeat_half_life", 30.0))
    currency = config.get("currency") or "USD"

    _empty_order_columns = [
        "id", "order_number", "checkout_id", "cart_token",
        "created_at", "updated_at", "processed_at", "cancelled_at", "closed_at",
        "financial_status", "fulfillment_status", "order_status_url", "confirmed", "test",
        "currency", "subtotal_price", "total_price", "total_tax", "total_discounts",
        "total_shipping_price", "total_refunded",
        "customer_id", "email", "source_name", "landing_site", "referring_site",
        "tags", "note", "shipping_address_country", "shipping_address_region",
        "billing_address_country", "billing_address_region", "billing_address_city",
        "last_attributed_ad_id", "last_attributed_adset_id", "last_attributed_campaign_id", "last_attributed_creative_id",
    ]
    _empty_line_item_columns = [
        "id", "order_id", "product_id", "variant_id", "sku",
        "product_title", "variant_title", "name", "vendor",
        "quantity", "fulfillable_quantity", "fulfillment_status",
        "price", "price_set_presentment_amount", "price_set_shop_amount",
        "total_discount", "discount_allocations", "taxable",
        "requires_shipping", "tax_lines", "created_at", "updated_at",
        "gift_card", "properties", "origin_location_id",
    ]
    _empty_transaction_columns = [
        "id", "order_id", "gateway", "kind", "status", "test", "currency", "amount",
        "authorization_code", "created_at", "processed_at", "updated_at",
        "source_name", "receipt", "error_code", "gateway_reference", "admin_graphql_api_id",
    ]

    # Only customers who have ordered before
    if "last_order_date" not in customers_df.columns:
        return (
            pd.DataFrame(columns=_empty_order_columns),
            pd.DataFrame(columns=_empty_line_item_columns),
            pd.DataFrame(columns=_empty_transaction_columns),
        )
    eligible = customers_df[pd.notna(customers_df["last_order_date"])].copy()
    if eligible.empty:
        return (
            pd.DataFrame(columns=_empty_order_columns),
            pd.DataFrame(columns=_empty_line_item_columns),
            pd.DataFrame(columns=_empty_transaction_columns),
        )

    current_ts = pd.Timestamp(current_date).normalize()
    eligible["_days_since"] = (current_ts - pd.to_datetime(eligible["last_order_date"]).dt.normalize()).dt.days
    eligible["_trust"] = eligible["trust_score"].fillna(0.5).values
    eligible["_sat"] = eligible["satisfaction_memory"].fillna(0).values
    eligible["_dis"] = eligible["disappointment_memory"].fillna(0).values
    # Tier 5: early disappointment (recent_negative_velocity) reduces repeat probability
    fragility_weight = float(config.get("repeat_fragility_weight", 0.2))
    if "recent_negative_velocity" in eligible.columns:
        eligible["_vel"] = eligible["recent_negative_velocity"].fillna(0).values
    else:
        eligible["_vel"] = 0.0
    # Price psychology: discount dependency hurts full-price repeat (no discount on repeat orders)
    dependency_penalty = float(config.get("dependency_repeat_penalty", 0.25))
    if "discount_dependency" in eligible.columns:
        eligible["_dep"] = eligible["discount_dependency"].fillna(0).values
    else:
        eligible["_dep"] = 0.0
    # Upgrade 6: shipping bad experience permanent penalty
    bad_ship_penalty = float(config.get("memory", {}).get("bad_shipping_permanent_penalty", 0.15))
    if "shipping_bad_count" in eligible.columns:
        eligible["_ship_bad"] = eligible["shipping_bad_count"].fillna(0).values
    else:
        eligible["_ship_bad"] = 0.0
    # Upgrade 6: discount-only buyers penalized on repeat (no full-price willingness)
    discount_only_penalty = float(config.get("memory", {}).get("discount_only_fullprice_penalty", 0.4))
    if "discount_only_buyer" in eligible.columns:
        eligible["_disc_only"] = eligible["discount_only_buyer"].fillna(False).astype(float).values
    else:
        eligible["_disc_only"] = 0.0
    recency_decay = np.exp(-np.maximum(eligible["_days_since"].values, 0) / half_life)
    p_repeat = (
        base_repeat
        * (0.5 + eligible["_trust"].values)
        * (1.0 + eligible["_sat"].values)
        * np.exp(-eligible["_dis"].values)
        * np.exp(-eligible["_vel"].values * fragility_weight)
        * np.exp(-eligible["_dep"].values * dependency_penalty)
        * recency_decay
        * np.where(eligible["_ship_bad"].values >= 2, 1.0 - bad_ship_penalty, 1.0)
        * np.where(eligible["_disc_only"].values > 0.5, 1.0 - discount_only_penalty, 1.0)
    )
    p_repeat = np.clip(p_repeat, 0.0, 0.5)
    repeat_draw = rng.random(size=len(eligible)) < p_repeat
    repeat_customers = eligible.loc[repeat_draw].copy()

    if repeat_customers.empty:
        return (
            pd.DataFrame(columns=_empty_order_columns),
            pd.DataFrame(columns=_empty_line_item_columns),
            pd.DataFrame(columns=_empty_transaction_columns),
        )

    n_repeat = len(repeat_customers)
    variant_list = variants_df.to_dict("records")
    n_variants = len(variant_list)
    if n_variants == 0:
        return (
            pd.DataFrame(columns=_empty_order_columns),
            pd.DataFrame(columns=_empty_line_item_columns),
            pd.DataFrame(columns=_empty_transaction_columns),
        )

    orders_rows: List[dict] = []
    line_items_rows: List[dict] = []
    transactions_rows: List[dict] = []

    for _, row in repeat_customers.iterrows():
        order_id = str(uuid.uuid4())
        customer_id = row["customer_id"]
        order_timestamp = current_ts + pd.Timedelta(minutes=int(rng.integers(1, 31)))

        order_row = {
            "order_id": order_id,
            "customer_id": customer_id,
            "order_timestamp": order_timestamp,
            "last_attributed_ad_id": row.get("last_attributed_ad_id"),
            "last_attributed_adset_id": row.get("last_attributed_adset_id"),
            "last_attributed_campaign_id": row.get("last_attributed_campaign_id"),
            "last_attributed_creative_id": row.get("last_attributed_creative_id"),
            "discount_pct": 0.0,
            "discount_code": None,
            "hesitation_proxy": pd.NA,
        }
        # Order address enrichment: customer default as anchor
        ship_country, ship_region, bill_country, bill_region = _enrich_order_addresses(
            row.get("default_address_country"),
            row.get("default_address_region"),
            rng,
        )
        order_row["shipping_address_country"] = ship_country
        order_row["shipping_address_region"] = ship_region
        order_row["billing_address_country"] = bill_country
        order_row["billing_address_region"] = bill_region
        order_row["billing_address_city"] = _pick_city_for_country(bill_country, rng)
        orders_rows.append(order_row)

        q_roll = rng.random()
        quantity = 1 if q_roll < 0.75 else rng.integers(2, 4)
        variant = variant_list[rng.integers(0, n_variants)]
        line_items_rows.append({
            "line_item_id": str(uuid.uuid4()),
            "order_id": order_id,
            "variant_id": variant["variant_id"],
            "quantity": quantity,
            "price": variant["price"],
        })
        transactions_rows.append({
            "transaction_id": str(uuid.uuid4()),
            "order_id": order_id,
            "status": "success",
        })

    orders_df = pd.DataFrame(orders_rows)
    line_items_df = pd.DataFrame(line_items_rows)
    transactions_df = pd.DataFrame(transactions_rows)

    # Enrich to same schema as simulate_purchases_from_clicks (typed nulls per contract)
    transactions_df = transactions_df.rename(columns={"transaction_id": "id"})
    for col in ["gateway", "kind", "authorization_code", "source_name", "receipt", "error_code", "gateway_reference", "admin_graphql_api_id"]:
        transactions_df[col] = None  # STRING (object)
    transactions_df["test"] = pd.array([pd.NA] * len(transactions_df), dtype="boolean")  # BOOL
    transactions_df["processed_at"] = pd.NaT  # TIMESTAMP
    transactions_df["updated_at"] = pd.NaT  # TIMESTAMP

    # Upgrade 7: spending_propensity multiplies repeat order line item prices
    if not orders_df.empty and "spending_propensity" in repeat_customers.columns:
        order_to_spend = dict(zip(
            orders_df["order_id"],
            [float(repeat_customers.iloc[i].get("spending_propensity", 1.0) or 1.0) for i in range(len(orders_df))]
        ))
        line_items_df = line_items_df.copy()
        line_items_df["price"] = line_items_df.apply(
            lambda r: round(float(r["price"]) * float(order_to_spend.get(r["order_id"], 1.0)), 2),
            axis=1,
        )

    line_items_df = line_items_df.copy()
    line_items_df["_line_total"] = line_items_df["price"] * line_items_df["quantity"]
    subtotal_by_order = line_items_df.groupby("order_id")["_line_total"].sum()
    line_items_df = line_items_df.drop(columns=["_line_total"])

    variant_cols = ["variant_id", "product_id"]
    if "title" in variants_df.columns:
        variant_cols.append("title")
    variants_lookup = variants_df[variant_cols].drop_duplicates("variant_id")
    if "title" in variants_lookup.columns:
        variants_lookup = variants_lookup.rename(columns={"title": "variant_title"})
    line_items_df = line_items_df.merge(variants_lookup, on="variant_id", how="left")
    if "variant_title" not in line_items_df.columns:
        line_items_df["variant_title"] = None
    if "product_id" in products_df.columns and "title" in products_df.columns:
        product_lookup = products_df[["product_id", "title"]].drop_duplicates("product_id").rename(
            columns={"title": "product_title"}
        )
        line_items_df = line_items_df.merge(product_lookup, on="product_id", how="left")
    if "product_title" not in line_items_df.columns:
        line_items_df["product_title"] = None
    line_items_df = line_items_df.merge(
        orders_df[["order_id", "order_timestamp"]],
        on="order_id",
        how="left",
    )
    line_items_df["created_at"] = line_items_df["order_timestamp"]
    line_items_df["updated_at"] = line_items_df["order_timestamp"]
    line_items_df = line_items_df.drop(columns=["order_timestamp"])
    line_items_df["fulfillable_quantity"] = line_items_df["quantity"]
    line_items_df["fulfillment_status"] = "fulfilled"
    line_items_df["taxable"] = True
    line_items_df["requires_shipping"] = True
    line_items_df["gift_card"] = False
    for col in ["sku", "vendor", "name", "discount_allocations", "tax_lines", "properties", "origin_location_id"]:
        line_items_df[col] = None  # STRING (object)
    line_items_df["price_set_presentment_amount"] = np.nan  # NUMERIC
    line_items_df["price_set_shop_amount"] = np.nan  # NUMERIC
    line_items_df["total_discount"] = np.nan  # NUMERIC
    line_items_df = line_items_df.rename(columns={"line_item_id": "id"})
    line_items_df = line_items_df[
        [
            "id", "order_id", "product_id", "variant_id", "sku",
            "product_title", "variant_title", "vendor", "name",
            "quantity", "fulfillable_quantity", "fulfillment_status",
            "price", "price_set_presentment_amount", "price_set_shop_amount",
            "total_discount", "discount_allocations", "taxable",
            "requires_shipping", "tax_lines",
            "created_at", "updated_at",
            "gift_card", "properties", "origin_location_id",
        ]
    ]

    orders_df["subtotal_price"] = orders_df["order_id"].map(subtotal_by_order).astype(float)
    orders_df["total_discounts"] = 0.0
    orders_df["total_tax"] = (orders_df["subtotal_price"] * 0.08).round(2)
    orders_df["total_shipping_price"] = 5.0
    orders_df["total_price"] = (
        orders_df["subtotal_price"]
        + orders_df["total_tax"]
        + orders_df["total_shipping_price"]
    ).round(2)
    orders_df["total_refunded"] = 0.0
    orders_df["currency"] = currency
    orders_df["financial_status"] = "paid"
    orders_df["fulfillment_status"] = "fulfilled"
    orders_df["confirmed"] = True
    orders_df["test"] = False
    orders_df = orders_df.rename(columns={"order_id": "id", "order_timestamp": "created_at"})
    orders_df["updated_at"] = orders_df["created_at"]
    orders_df["processed_at"] = orders_df["created_at"]

    order_for_tx = orders_df[["id", "currency", "total_price", "created_at"]].rename(
        columns={"id": "order_id", "total_price": "amount"}
    )
    transactions_df = transactions_df.merge(order_for_tx, on="order_id", how="left")
    transactions_df = transactions_df[
        [
            "id", "order_id", "gateway", "kind", "status", "test", "currency", "amount",
            "authorization_code", "created_at", "processed_at", "updated_at",
            "source_name", "receipt", "error_code", "gateway_reference", "admin_graphql_api_id",
        ]
    ]

    if "email" in customers_df.columns:
        customer_email = customers_df[["customer_id", "email"]].drop_duplicates("customer_id")
        orders_df = orders_df.merge(customer_email, on="customer_id", how="left")
    else:
        orders_df["email"] = None
    for col in ["cancelled_at", "closed_at"]:
        if col not in orders_df.columns:
            orders_df[col] = pd.NaT  # TIMESTAMP
    for col in [
        "order_number", "checkout_id", "cart_token", "order_status_url", "source_name",
        "landing_site", "referring_site", "tags", "note",
        "shipping_address_country", "shipping_address_region",
        "billing_address_country", "billing_address_region", "billing_address_city",
    ]:
        if col not in orders_df.columns:
            orders_df[col] = None  # STRING (object)
    repeat_out_cols = [
        "id", "order_number", "checkout_id", "cart_token",
        "created_at", "updated_at", "processed_at", "cancelled_at", "closed_at",
        "financial_status", "fulfillment_status", "order_status_url", "confirmed", "test",
        "currency", "subtotal_price", "total_price", "total_tax", "total_discounts",
        "total_shipping_price", "total_refunded",
        "customer_id", "email", "source_name", "landing_site", "referring_site",
        "tags", "note", "shipping_address_country", "shipping_address_region",
        "billing_address_country", "billing_address_region", "billing_address_city",
        "last_attributed_ad_id", "last_attributed_adset_id", "last_attributed_campaign_id", "last_attributed_creative_id",
    ]
    for c in ["discount_pct", "discount_code", "hesitation_proxy"]:
        if c in orders_df.columns and c not in repeat_out_cols:
            repeat_out_cols.append(c)
    orders_df = orders_df[[c for c in repeat_out_cols if c in orders_df.columns]]
    orders_df["order_id"] = orders_df["id"].values
    orders_df["order_timestamp"] = orders_df["created_at"].values

    return (orders_df, line_items_df, transactions_df)


def simulate_cart_returns(
    candidates_df: pd.DataFrame,
    variants_df: pd.DataFrame,
    products_df: pd.DataFrame,
    config: dict,
    current_date: pd.Timestamp,
    creatives_df: "pd.DataFrame | None" = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan humans with active cart_memory; compute causal return probability; resume at checkout.

    Returns (orders_df, line_items_df, transactions_df, new_customers_df, checkouts_df, funnel_events_df).
    """
    currency = config.get("currency", "USD")
    cart_return_half_life = config.get("cart_return_half_life", 5.0)
    seed = config.get("seed", 42)
    start_date = pd.Timestamp(config.get("start_date", "2023-01-01"))
    day_ordinal = (current_date - start_date).days
    rng = np.random.default_rng(seed + day_ordinal + 7777)

    _empty_order_columns = [
        "id", "order_number", "checkout_id", "cart_token",
        "created_at", "updated_at", "processed_at", "cancelled_at", "closed_at",
        "financial_status", "fulfillment_status", "order_status_url", "confirmed", "test",
        "currency", "subtotal_price", "total_price", "total_tax", "total_discounts",
        "total_shipping_price", "total_refunded",
        "customer_id", "email", "source_name", "landing_site", "referring_site",
        "tags", "note", "shipping_address_country", "shipping_address_region",
        "billing_address_country", "billing_address_region", "billing_address_city",
        "last_attributed_ad_id", "last_attributed_adset_id", "last_attributed_campaign_id", "last_attributed_creative_id",
    ]
    _empty_line_item_columns = [
        "id", "order_id", "product_id", "variant_id", "sku",
        "product_title", "variant_title", "vendor", "name",
        "quantity", "fulfillable_quantity", "fulfillment_status",
        "price", "price_set_presentment_amount", "price_set_shop_amount",
        "total_discount", "discount_allocations", "taxable",
        "requires_shipping", "tax_lines", "created_at", "updated_at",
        "gift_card", "properties", "origin_location_id",
    ]
    _empty_transaction_columns = [
        "id", "order_id", "gateway", "kind", "status", "test", "currency", "amount",
        "authorization_code", "created_at", "processed_at", "updated_at",
        "source_name", "receipt", "error_code", "gateway_reference", "admin_graphql_api_id",
    ]
    _empty_checkout_columns = [
        "id", "token", "cart_token", "customer_id",
        "created_at", "updated_at", "completed_at", "closed_at",
        "currency", "total_price", "subtotal_price", "total_tax", "total_discounts", "total_line_items_price",
        "source_name", "landing_site", "referring_site", "abandoned_checkout_url", "email",
    ]
    _empty_funnel_columns = ["ad_id", "adset_id", "campaign_id", "creative_id", "customer_id", "date", "event_type"]
    _empty_customers_columns = ["id", "customer_id", "created_at"]

    def _empty_result():
        return (
            pd.DataFrame(columns=_empty_order_columns),
            pd.DataFrame(columns=_empty_line_item_columns),
            pd.DataFrame(columns=_empty_transaction_columns),
            pd.DataFrame(columns=_empty_customers_columns),
            pd.DataFrame(columns=_empty_checkout_columns),
            pd.DataFrame(columns=_empty_funnel_columns),
        )

    if candidates_df is None or candidates_df.empty:
        return _empty_result()

    has_cart = candidates_df["cart_memory"].apply(lambda x: x is not None and isinstance(x, dict))
    cart_humans = candidates_df.loc[has_cart].copy()
    if cart_humans.empty:
        return _empty_result()

    # Extract cart fields into columns for vectorized operations
    cart_humans["_cart_age"] = cart_humans["cart_memory"].apply(lambda m: m.get("cart_age_days", 0))
    cart_humans["_cart_alignment"] = cart_humans["cart_memory"].apply(lambda m: m.get("alignment", 0.5))
    cart_humans["_cart_discount_pct"] = cart_humans["cart_memory"].apply(lambda m: m.get("discount_pct", 0.0))
    cart_humans["_cart_ad_id"] = cart_humans["cart_memory"].apply(lambda m: m.get("ad_id"))
    cart_humans["_cart_adset_id"] = cart_humans["cart_memory"].apply(lambda m: m.get("adset_id"))
    cart_humans["_cart_campaign_id"] = cart_humans["cart_memory"].apply(lambda m: m.get("campaign_id"))
    cart_humans["_cart_creative_id"] = cart_humans["cart_memory"].apply(lambda m: m.get("creative_id"))
    cart_humans["_cart_discount_code"] = cart_humans["cart_memory"].apply(lambda m: m.get("discount_code"))

    trust = cart_humans["trust_score"].fillna(0.5).values
    alignment = cart_humans["_cart_alignment"].values
    discount_dep = cart_humans["discount_dependency"].fillna(0.0).values
    cart_discount = cart_humans["_cart_discount_pct"].fillna(0.0).values
    neg_vel = cart_humans["recent_negative_velocity"].fillna(0.0).values
    disappointment = cart_humans["disappointment_memory"].fillna(0.0).values
    cart_age = cart_humans["_cart_age"].values.astype(float)

    # Fatigue to original creative (if tracking exists)
    fatigue = np.zeros(len(cart_humans))
    if "creative_fatigue_map" in cart_humans.columns:
        for i, (_, h) in enumerate(cart_humans.iterrows()):
            fmap = h.get("creative_fatigue_map")
            cid = h.get("_cart_creative_id")
            if isinstance(fmap, dict) and cid is not None:
                fatigue[i] = fmap.get(str(cid), 0.0)

    # Causal return probability
    time_decay = np.exp(-cart_age / cart_return_half_life)
    discount_penalty = np.where(discount_dep > 0.5, np.where(cart_discount > 0, 0.0, discount_dep * 0.3), 0.0)
    p_return = (
        0.10
        + 0.25 * trust
        + 0.20 * alignment
        - 0.15 * neg_vel
        - 0.10 * disappointment
        - 0.10 * fatigue
        - discount_penalty
    ) * time_decay
    p_return = np.clip(p_return, 0.01, 0.60)

    roll = rng.random(len(cart_humans))
    returning_mask = roll < p_return
    returning = cart_humans.loc[returning_mask].copy()

    if returning.empty:
        return _empty_result()

    # Checkout conversion probability (same model as funnel step 4)
    r_trust = returning["trust_score"].fillna(0.5).values
    r_alignment = returning["_cart_alignment"].values
    r_fatigue = np.zeros(len(returning))
    if "creative_fatigue_map" in returning.columns:
        for i, (_, h) in enumerate(returning.iterrows()):
            fmap = h.get("creative_fatigue_map")
            cid = h.get("_cart_creative_id")
            if isinstance(fmap, dict) and cid is not None:
                r_fatigue[i] = fmap.get(str(cid), 0.0)
    r_neg_vel = returning["recent_negative_velocity"].fillna(0.0).values
    r_disappointment = returning["disappointment_memory"].fillna(0.0).values

    p_purchase = (
        0.20
        + 0.30 * r_trust
        + 0.15 * r_alignment
        - 0.10 * r_fatigue
        - 0.10 * r_neg_vel
        - 0.08 * r_disappointment
    )
    p_purchase = np.clip(p_purchase, 0.05, 0.80)

    p_roll = rng.random(len(returning))
    purchased_mask = p_roll < p_purchase
    purchased = returning.loc[purchased_mask].copy()

    if purchased.empty:
        # Emit initiate_checkout funnel events for returning humans (even if none purchase)
        fe_rows = []
        for _, h in returning.iterrows():
            fe_rows.append({
                "ad_id": h["_cart_ad_id"],
                "adset_id": h["_cart_adset_id"],
                "campaign_id": h["_cart_campaign_id"],
                "creative_id": h["_cart_creative_id"],
                "customer_id": h.get("customer_id", h.get("id")),
                "date": current_date.strftime("%Y-%m-%d"),
                "event_type": "initiate_checkout",
            })
        funnel_events_df = pd.DataFrame(fe_rows) if fe_rows else pd.DataFrame(columns=_empty_funnel_columns)

        # Abandoned checkouts for returning humans
        co_rows = []
        for _, h in returning.iterrows():
            co_rows.append({
                "id": str(uuid.uuid4()),
                "token": str(uuid.uuid4()),
                "cart_token": str(uuid.uuid4()),
                "customer_id": h.get("customer_id", h.get("id")),
                "created_at": current_date,
                "updated_at": current_date,
                "completed_at": pd.NaT,
                "closed_at": pd.NaT,
                "currency": currency,
                "total_price": 0.0,
                "subtotal_price": 0.0,
                "total_tax": 0.0,
                "total_discounts": 0.0,
                "total_line_items_price": 0.0,
                "source_name": "meta",
                "landing_site": None,
                "referring_site": None,
                "abandoned_checkout_url": "https://store/checkout/recovery",
                "email": None,
            })
        checkouts_df = pd.DataFrame(co_rows) if co_rows else pd.DataFrame(columns=_empty_checkout_columns)
        return (
            pd.DataFrame(columns=_empty_order_columns),
            pd.DataFrame(columns=_empty_line_item_columns),
            pd.DataFrame(columns=_empty_transaction_columns),
            pd.DataFrame(columns=_empty_customers_columns),
            checkouts_df,
            funnel_events_df,
        )

    # Build orders, line items, transactions for purchasing humans
    variant_list = variants_df.to_dict("records")
    n_variants = len(variant_list)
    if n_variants == 0:
        return _empty_result()

    orders_rows: List[dict] = []
    line_items_rows: List[dict] = []
    transactions_rows: List[dict] = []
    new_customers_rows: List[dict] = []
    prospect_to_customer: dict = {}

    for _, row in purchased.iterrows():
        order_id = str(uuid.uuid4())
        raw_id = str(row.get("customer_id", row.get("id")))
        is_prospect = raw_id.startswith("prospect_")

        if is_prospect:
            if raw_id in prospect_to_customer:
                cust_id = prospect_to_customer[raw_id]
            else:
                cust_id = "cust_" + raw_id
                prospect_to_customer[raw_id] = cust_id
                signup_date = current_date.normalize()
                nc = {
                    "id": cust_id,
                    "customer_id": cust_id,
                    "created_at": signup_date,
                    "updated_at": signup_date,
                    "signup_date": signup_date,
                    "price_sensitivity": row.get("price_sensitivity", 0.5),
                    "impulse_level": row.get("impulse_level", 0.5),
                    "loyalty_propensity": row.get("loyalty_propensity", 0.5),
                    "regret_propensity": row.get("regret_propensity", 0.5),
                    "quality_expectation": row.get("quality_expectation", 0.5),
                    "income_proxy": row.get("income_proxy", "mid"),
                    "acquisition_channel_preference": row.get("acquisition_channel_preference", "meta"),
                    "trust_score": 0.5,
                    "disappointment_memory": 0.0,
                    "satisfaction_memory": 0.0,
                    "recent_negative_velocity": 0.0,
                    "discount_dependency": 0.0,
                    "exposure_count": 0,
                    "expressed_desire_level": 0.1,
                    "desire_decay_memory": 0.0,
                    "creative_fatigue_map": {},
                    "cart_memory": None,
                    "last_attributed_ad_id": row.get("_cart_ad_id"),
                    "last_attributed_adset_id": row.get("_cart_adset_id"),
                    "last_attributed_campaign_id": row.get("_cart_campaign_id"),
                    "last_attributed_creative_id": row.get("_cart_creative_id"),
                    "last_exposure_date": pd.NaT,
                    "last_order_date": pd.NaT,
                    "last_order_at": pd.NaT,
                    "first_touch_ad_id": row.get("_cart_ad_id"),
                    "first_touch_adset_id": row.get("_cart_adset_id"),
                    "first_touch_campaign_id": row.get("_cart_campaign_id"),
                    "first_touch_creative_id": row.get("_cart_creative_id"),
                    "first_touch_date": signup_date,
                    "state": row.get("state", "enabled"),
                    "verified_email": row.get("verified_email", True),
                    "email": row.get("email"),
                    "first_name": row.get("first_name"),
                    "last_name": row.get("last_name"),
                    "accepts_marketing": row.get("accepts_marketing", False),
                    "default_address_country": row.get("default_address_country"),
                    "default_address_region": row.get("default_address_region"),
                    "default_address_city": row.get("default_address_city"),
                }
                for col in ["phone", "locale", "tags", "note", "source_name", "display_name", "default_address_id", "default_address_postal_code"]:
                    nc[col] = row.get(col)
                nc["tax_exempt"] = pd.NA
                new_customers_rows.append(nc)
        else:
            cust_id = raw_id

        hesitation_minutes = int(row["_cart_age"]) * 1440 + rng.integers(0, 120)
        order_timestamp = current_date + pd.Timedelta(minutes=rng.integers(0, 720))

        ship_country, ship_region, bill_country, bill_region = _enrich_order_addresses(
            row.get("default_address_country"), row.get("default_address_region"), rng
        )

        order_row = {
            "order_id": order_id,
            "customer_id": cust_id,
            "order_timestamp": order_timestamp,
            "hesitation_proxy": hesitation_minutes,
            "last_attributed_ad_id": row["_cart_ad_id"],
            "last_attributed_adset_id": row["_cart_adset_id"],
            "last_attributed_campaign_id": row["_cart_campaign_id"],
            "last_attributed_creative_id": row["_cart_creative_id"],
            "discount_pct": row["_cart_discount_pct"],
            "discount_code": row["_cart_discount_code"],
            "shipping_address_country": ship_country,
            "shipping_address_region": ship_region,
            "billing_address_country": bill_country,
            "billing_address_region": bill_region,
            "billing_address_city": _pick_city_for_country(bill_country, rng),
        }
        orders_rows.append(order_row)

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

    orders_df = pd.DataFrame(orders_rows)
    line_items_df = pd.DataFrame(line_items_rows)
    transactions_df = pd.DataFrame(transactions_rows)
    new_customers_df = pd.DataFrame(new_customers_rows) if new_customers_rows else pd.DataFrame(columns=_empty_customers_columns)

    # Enrich transactions
    transactions_df = transactions_df.rename(columns={"transaction_id": "id"})
    for col in ["gateway", "kind", "authorization_code", "source_name", "receipt", "error_code", "gateway_reference", "admin_graphql_api_id"]:
        transactions_df[col] = None
    transactions_df["test"] = pd.array([pd.NA] * len(transactions_df), dtype="boolean")
    transactions_df["processed_at"] = pd.NaT
    transactions_df["updated_at"] = pd.NaT

    # Subtotal and enrich line items
    line_items_df = line_items_df.copy()
    line_items_df["_line_total"] = line_items_df["price"] * line_items_df["quantity"]
    subtotal_by_order = line_items_df.groupby("order_id")["_line_total"].sum()
    line_items_df = line_items_df.drop(columns=["_line_total"])

    variant_cols = ["variant_id", "product_id"]
    if "title" in variants_df.columns:
        variant_cols.append("title")
    variants_lookup = variants_df[variant_cols].drop_duplicates("variant_id")
    if "title" in variants_lookup.columns:
        variants_lookup = variants_lookup.rename(columns={"title": "variant_title"})
    line_items_df = line_items_df.merge(variants_lookup, on="variant_id", how="left")
    if "variant_title" not in line_items_df.columns:
        line_items_df["variant_title"] = None
    if "product_id" in products_df.columns and "title" in products_df.columns:
        product_lookup = products_df[["product_id", "title"]].drop_duplicates("product_id").rename(columns={"title": "product_title"})
        line_items_df = line_items_df.merge(product_lookup, on="product_id", how="left")
    if "product_title" not in line_items_df.columns:
        line_items_df["product_title"] = None
    line_items_df = line_items_df.merge(orders_df[["order_id", "order_timestamp"]], on="order_id", how="left")
    line_items_df["created_at"] = line_items_df["order_timestamp"]
    line_items_df["updated_at"] = line_items_df["order_timestamp"]
    line_items_df = line_items_df.drop(columns=["order_timestamp"])
    line_items_df["fulfillable_quantity"] = line_items_df["quantity"]
    line_items_df["fulfillment_status"] = "fulfilled"
    line_items_df["taxable"] = True
    line_items_df["requires_shipping"] = True
    line_items_df["gift_card"] = False
    for col in ["sku", "vendor", "name", "discount_allocations", "tax_lines", "properties", "origin_location_id"]:
        line_items_df[col] = None
    line_items_df["price_set_presentment_amount"] = np.nan
    line_items_df["price_set_shop_amount"] = np.nan
    line_items_df["total_discount"] = np.nan
    line_items_df = line_items_df.rename(columns={"line_item_id": "id"})
    li_out = [
        "id", "order_id", "product_id", "variant_id", "sku",
        "product_title", "variant_title", "vendor", "name",
        "quantity", "fulfillable_quantity", "fulfillment_status",
        "price", "price_set_presentment_amount", "price_set_shop_amount",
        "total_discount", "discount_allocations", "taxable",
        "requires_shipping", "tax_lines", "created_at", "updated_at",
        "gift_card", "properties", "origin_location_id",
    ]
    line_items_df = line_items_df[[c for c in li_out if c in line_items_df.columns]]

    # Order totals
    orders_df["subtotal_price"] = orders_df["order_id"].map(subtotal_by_order).astype(float)
    orders_df["discount_pct"] = orders_df.get("discount_pct", 0.0)
    if isinstance(orders_df["discount_pct"], pd.Series):
        orders_df["discount_pct"] = orders_df["discount_pct"].fillna(0)
    orders_df["total_discounts"] = (orders_df["subtotal_price"] * orders_df["discount_pct"]).round(2)
    orders_df["total_tax"] = ((orders_df["subtotal_price"] - orders_df["total_discounts"]) * 0.08).round(2)
    orders_df["total_shipping_price"] = 5.0
    orders_df["total_price"] = (
        orders_df["subtotal_price"] - orders_df["total_discounts"] + orders_df["total_tax"] + orders_df["total_shipping_price"]
    ).round(2)
    orders_df["total_refunded"] = 0.0
    if "discount_code" not in orders_df.columns:
        orders_df["discount_code"] = None
    orders_df["currency"] = currency
    orders_df["financial_status"] = "paid"
    orders_df["fulfillment_status"] = "fulfilled"
    orders_df["confirmed"] = True
    orders_df["test"] = False

    orders_df = orders_df.rename(columns={"order_id": "id", "order_timestamp": "created_at"})
    orders_df["updated_at"] = orders_df["created_at"]
    orders_df["processed_at"] = orders_df["created_at"]
    for col in ["cancelled_at", "closed_at"]:
        if col not in orders_df.columns:
            orders_df[col] = pd.NaT
    for col in [
        "order_number", "checkout_id", "cart_token", "order_status_url", "source_name",
        "landing_site", "referring_site", "tags", "note", "email",
    ]:
        if col not in orders_df.columns:
            orders_df[col] = None
    for col in ["shipping_address_country", "shipping_address_region", "billing_address_country", "billing_address_region", "billing_address_city"]:
        if col not in orders_df.columns:
            orders_df[col] = None

    out_cols = [
        "id", "order_number", "checkout_id", "cart_token",
        "created_at", "updated_at", "processed_at", "cancelled_at", "closed_at",
        "financial_status", "fulfillment_status", "order_status_url", "confirmed", "test",
        "currency", "subtotal_price", "total_price", "total_tax", "total_discounts",
        "total_shipping_price", "total_refunded",
        "customer_id", "email", "source_name", "landing_site", "referring_site",
        "tags", "note", "shipping_address_country", "shipping_address_region",
        "billing_address_country", "billing_address_region", "billing_address_city",
        "last_attributed_ad_id", "last_attributed_adset_id", "last_attributed_campaign_id", "last_attributed_creative_id",
    ]
    for c in ["discount_pct", "discount_code", "hesitation_proxy"]:
        if c in orders_df.columns and c not in out_cols:
            out_cols.append(c)
    orders_df = orders_df[[c for c in out_cols if c in orders_df.columns]]
    orders_df["order_id"] = orders_df["id"].values
    orders_df["order_timestamp"] = orders_df["created_at"].values

    # Transactions enrichment
    order_for_tx = orders_df[["id", "currency", "total_price", "created_at"]].rename(
        columns={"id": "order_id", "total_price": "amount"}
    )
    transactions_df = transactions_df.merge(order_for_tx, on="order_id", how="left")
    tx_cols = [
        "id", "order_id", "gateway", "kind", "status", "test", "currency", "amount",
        "authorization_code", "created_at", "processed_at", "updated_at",
        "source_name", "receipt", "error_code", "gateway_reference", "admin_graphql_api_id",
    ]
    transactions_df = transactions_df[[c for c in tx_cols if c in transactions_df.columns]]

    # Checkouts for all returning humans (purchased = completed, others = abandoned)
    purchased_ids = set(purchased.get("customer_id", purchased.get("id")).astype(str))
    co_rows = []
    for _, h in returning.iterrows():
        cid = h.get("customer_id", h.get("id"))
        is_purchased = str(cid) in purchased_ids
        matching_order = orders_df.loc[orders_df["customer_id"].astype(str) == str(cid)]
        checkout_ts = current_date + pd.Timedelta(minutes=rng.integers(0, 60))
        if is_purchased and not matching_order.empty:
            orow = matching_order.iloc[0]
            co_rows.append({
                "id": str(uuid.uuid4()), "token": str(uuid.uuid4()), "cart_token": str(uuid.uuid4()),
                "customer_id": cid,
                "created_at": checkout_ts, "updated_at": orow["created_at"],
                "completed_at": orow["created_at"], "closed_at": orow["created_at"],
                "currency": currency,
                "total_price": float(orow["total_price"]), "subtotal_price": float(orow["subtotal_price"]),
                "total_tax": float(orow["total_tax"]), "total_discounts": float(orow["total_discounts"]),
                "total_line_items_price": float(orow["subtotal_price"]),
                "source_name": "meta", "landing_site": None, "referring_site": None,
                "abandoned_checkout_url": None, "email": None,
            })
        else:
            co_rows.append({
                "id": str(uuid.uuid4()), "token": str(uuid.uuid4()), "cart_token": str(uuid.uuid4()),
                "customer_id": cid,
                "created_at": checkout_ts, "updated_at": checkout_ts,
                "completed_at": pd.NaT, "closed_at": pd.NaT,
                "currency": currency,
                "total_price": 0.0, "subtotal_price": 0.0,
                "total_tax": 0.0, "total_discounts": 0.0, "total_line_items_price": 0.0,
                "source_name": "meta", "landing_site": None, "referring_site": None,
                "abandoned_checkout_url": "https://store/checkout/recovery", "email": None,
            })
    checkouts_df = pd.DataFrame(co_rows) if co_rows else pd.DataFrame(columns=_empty_checkout_columns)

    # Funnel events: all returning humans emit initiate_checkout
    fe_rows = []
    for _, h in returning.iterrows():
        fe_rows.append({
            "ad_id": h["_cart_ad_id"],
            "adset_id": h["_cart_adset_id"],
            "campaign_id": h["_cart_campaign_id"],
            "creative_id": h["_cart_creative_id"],
            "customer_id": h.get("customer_id", h.get("id")),
            "date": current_date.strftime("%Y-%m-%d"),
            "event_type": "initiate_checkout",
        })
    funnel_events_df = pd.DataFrame(fe_rows) if fe_rows else pd.DataFrame(columns=_empty_funnel_columns)

    return (orders_df, line_items_df, transactions_df, new_customers_df, checkouts_df, funnel_events_df)
