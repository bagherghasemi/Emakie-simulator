"""
Customer population generator for behavioral e-commerce simulation.

Produces a DataFrame of synthetic customers with stable latent traits and
behavioral anchors. These fields are hidden variables used later by simulation
mechanics (e.g. propensity to buy on impulse, sensitivity to price) and should
not be exposed as "observed" data—they drive event generation.

Output is wrapped in a Shopify-style schema (id, created_at, email, first_name,
etc.) so dbt can read it as shopify_customers; all behavioral traits are kept.
"""

import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Simple lists for synthetic identity (realism without complexity)
FIRST_NAMES = [
    "Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley",
    "Jamie", "Quinn", "Avery", "Reese", "Skyler", "Cameron", "Drew",
    "Blake", "Finley", "Parker", "Hayden", "Emery", "Rowan",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
]
# Simple fake regions and cities for default address
FAKE_REGIONS = ["North", "South", "East", "West", "Central", "Metro", "Coastal"]
FAKE_CITIES = ["Springfield", "Riverside", "Oakdale", "Lakeside", "Hillview", "Brookside", "Fairview"]


def generate_customers(config: dict) -> pd.DataFrame:
    """
    Generate a population of customers with stable latent traits for simulation.

    Reads simulation start_date, end_date, and number of customers from config.
    Optionally uses config["seed"] for numpy RNG reproducibility.

    Each customer has:
    - Identity: customer_id (UUID), signup_date (between start and end).
    - Latent traits (floats in [0, 1]): hidden variables that influence future
      behavior (e.g. price_sensitivity affects basket size and promotion response).
      Values are normally distributed then clamped to [0, 1]. Mild correlations
      are applied: higher impulse_level and quality_expectation slightly raise
      regret_propensity; higher loyalty_propensity slightly lowers it (then
      clamped again).
    - Behavioral anchors: income_proxy (low/mid/high, skewed toward mid) and
      acquisition_channel_preference (meta/organic/search).

    These latent traits and anchors are used by downstream simulation mechanics
    to drive events (clicks, carts, purchases, returns) in a consistent way
    over the simulated period.

    Parameters
    ----------
    config : dict
        Must contain:
        - simulation.start_date : str or date-like (e.g. "2022-01-01")
        - simulation.end_date : str or date-like (e.g. "2024-01-01")
        - number of customers : int (key may be "num_customers" or "customers")
        Optional:
        - seed : int for numpy random seed (reproducibility)

    Returns
    -------
    pandas.DataFrame
        One row per customer. Shopify-style columns: id, created_at, updated_at,
        email, first_name, last_name, state, verified_email, accepts_marketing,
        default_address_country/region/city; NULL: phone, locale, last_order_at,
        tags, note, source_name, etc. Behavioral traits unchanged: price_sensitivity,
        impulse_level, loyalty_propensity, regret_propensity, quality_expectation,
        income_proxy, acquisition_channel_preference.
        customer_id and signup_date are present as aliases for downstream simulation.
    """
    # Reproducibility
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    # Config parsing
    sim = config.get("simulation", config)
    start_date = pd.to_datetime(sim.get("start_date"))
    end_date = pd.to_datetime(sim.get("end_date"))
    n = config.get("num_customers") or config.get("customers") or config.get("number_of_customers")
    if n is None:
        raise ValueError("config must specify number of customers (e.g. num_customers)")

    n = int(n)

    # Identity: unique UUIDs and signup dates evenly distributed in time
    customer_ids = [str(uuid.uuid4()) for _ in range(n)]
    start_ts = start_date.value
    end_ts = end_date.value
    signup_ts = rng.integers(start_ts, end_ts + 1, size=n)
    signup_dates = pd.to_datetime(signup_ts, unit="ns")

    # Latent traits: normal distribution, clamped to [0, 1]
    # Mean 0.5, std 0.2 gives most mass in (0.1, 0.9)
    mean, std = 0.5, 0.2
    latent = rng.normal(mean, std, size=(n, 5))
    latent = np.clip(latent, 0.0, 1.0)

    price_sensitivity = latent[:, 0].copy()
    impulse_level = latent[:, 1].copy()
    loyalty_propensity = latent[:, 2].copy()
    regret_propensity = latent[:, 3].copy()
    quality_expectation = latent[:, 4].copy()

    # Trait correlations: mild influence on regret_propensity, then reclamp
    # impulse ↑ → regret ↑; loyalty ↑ → regret ↓; quality ↑ → regret ↑
    regret_propensity = (
        regret_propensity
        + 0.12 * impulse_level
        - 0.12 * loyalty_propensity
        + 0.10 * quality_expectation
        + rng.normal(0, 0.05, size=n)
    )
    regret_propensity = np.clip(regret_propensity, 0.0, 1.0)

    # Behavioral anchors: income_proxy with realistic skew (more mid)
    income_levels = np.array(["low", "mid", "high"])
    income_probs = np.array([0.25, 0.55, 0.20])  # more mid
    income_proxy = rng.choice(income_levels, size=n, p=income_probs)

    # Acquisition channel preference
    channels = np.array(["meta", "organic", "search"])
    channel_probs = np.array([0.35, 0.35, 0.30])  # can be tuned
    acquisition_channel_preference = rng.choice(channels, size=n, p=channel_probs)

    # Upgrade 7: spending propensity — lognormal heterogeneity for basket size
    spending_sigma = float(config.get("distribution_realism", {}).get("spending_propensity_sigma", 0.5))
    spending_propensity = np.clip(rng.lognormal(0, spending_sigma, size=n), 0.2, 20.0)

    # Behavioral state (Tier 2): evolve after purchases via expectation/experience mismatch
    trust_score = np.full(n, 0.5)
    disappointment_memory = np.zeros(n)
    satisfaction_memory = np.zeros(n)

    # Tier 5: early disappointment → future churn signal (moving avg of recent positive mismatches)
    recent_negative_velocity = np.zeros(n)

    # Price psychology: discount dependency (promo-trained customers hesitate at full price; internal only)
    discount_dependency = np.zeros(n)

    # Tier 5: cumulative exposure count for drift accumulation (culture mutates over time)
    exposure_count = np.zeros(n, dtype=np.int64)

    # Expressed desire: motivational energy before purchase; accumulates from exposure/click, decays daily (denominator of yield)
    expressed_desire_level = rng.uniform(0.05, 0.15, size=n).astype(float)
    desire_decay_memory = np.zeros(n, dtype=float)

    # Tier 4 creative fatigue: per-customer per-creative exposure decay (simulation state only, not persisted)
    creative_fatigue_map = [{} for _ in range(n)]

    # Tier 6 persistent cart memory: active cart state per human (simulation state only, dropped before BQ)
    cart_memory = [None] * n

    # Tier 3 repeat: last order date and attribution (updated in main after each day's orders)
    last_attributed_ad_id = [None] * n
    last_attributed_adset_id = [None] * n
    last_attributed_campaign_id = [None] * n
    last_attributed_creative_id = [None] * n

    # Build DataFrame with Shopify-style schema (rename and enrich)
    df = pd.DataFrame({
        "id": customer_ids,
        "created_at": signup_dates,
        "price_sensitivity": price_sensitivity,
        "impulse_level": impulse_level,
        "loyalty_propensity": loyalty_propensity,
        "regret_propensity": regret_propensity,
        "quality_expectation": quality_expectation,
        "income_proxy": income_proxy,
        "acquisition_channel_preference": acquisition_channel_preference,
        "spending_propensity": spending_propensity,
        "trust_score": trust_score,
        "disappointment_memory": disappointment_memory,
        "satisfaction_memory": satisfaction_memory,
        "recent_negative_velocity": recent_negative_velocity,
        "discount_dependency": discount_dependency,
        "exposure_count": exposure_count,
        "expressed_desire_level": expressed_desire_level,
        "desire_decay_memory": desire_decay_memory,
        "creative_fatigue_map": creative_fatigue_map,
        "cart_memory": cart_memory,
        "last_attributed_ad_id": last_attributed_ad_id,
        "last_attributed_adset_id": last_attributed_adset_id,
        "last_attributed_campaign_id": last_attributed_campaign_id,
        "last_attributed_creative_id": last_attributed_creative_id,
        "last_exposure_date": [pd.NaT] * n,
    })
    df["last_order_date"] = pd.NaT

    df["updated_at"] = df["created_at"]
    df["state"] = "enabled"
    df["verified_email"] = True

    # Realistic synthetic identity
    df["email"] = "customer_" + df["id"].str[:8] + "@example.com"
    df["first_name"] = rng.choice(FIRST_NAMES, size=n).tolist()
    df["last_name"] = rng.choice(LAST_NAMES, size=n).tolist()

    # Marketing behavior
    df["accepts_marketing"] = rng.random(size=n) < 0.70

    # Default address (simple fake values)
    df["default_address_country"] = rng.choice(
        ["US", "CA", "UK", "DE"], size=n
    ).tolist()
    df["default_address_region"] = rng.choice(FAKE_REGIONS, size=n).tolist()
    df["default_address_city"] = rng.choice(FAKE_CITIES, size=n).tolist()

    # Typed nulls per contract: TIMESTAMP → pd.NaT, BOOL → nullable bool, STRING → None
    df["last_order_at"] = pd.NaT  # TIMESTAMP
    df["tax_exempt"] = pd.array([pd.NA] * n, dtype="boolean")  # BOOL
    for col in ["phone", "locale", "tags", "note", "source_name", "display_name", "default_address_id", "default_address_postal_code"]:
        df[col] = None  # STRING (object)

    # Aliases for downstream simulator (no behavioral change)
    df["customer_id"] = df["id"].values
    df["signup_date"] = df["created_at"].values

    return df


def generate_prospects(config: dict) -> pd.DataFrame:
    """
    Generate a pool of anonymous humans (prospects) who can be exposed to ads and click.
    First purchase creates the customer; prospects are never written to the warehouse.

    Same latent traits and address shape as customers so exposure/conversion logic
    works. Uses prospect_id as id/customer_id and signup_date = min date so they
    are always eligible for exposure. Deterministic given seed.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)
    sim = config.get("simulation", config)
    n = config.get("num_prospects") or config.get("num_customers") or config.get("customers")
    if n is None:
        raise ValueError("config must specify num_prospects or num_customers")
    n = int(n)

    # Deterministic prospect ids (same seed => same ids)
    prospect_ids = [f"prospect_{seed}_{i}" for i in range(n)]
    # Always eligible for exposure (no signup gate)
    signup_dates = [pd.Timestamp("1900-01-01")] * n

    mean, std = 0.5, 0.2
    latent = rng.normal(mean, std, size=(n, 5))
    latent = np.clip(latent, 0.0, 1.0)
    price_sensitivity = latent[:, 0].copy()
    impulse_level = latent[:, 1].copy()
    loyalty_propensity = latent[:, 2].copy()
    regret_propensity = latent[:, 3].copy()
    quality_expectation = latent[:, 4].copy()
    regret_propensity = np.clip(
        regret_propensity
        + 0.12 * impulse_level
        - 0.12 * loyalty_propensity
        + 0.10 * quality_expectation
        + rng.normal(0, 0.05, size=n),
        0.0,
        1.0,
    )
    income_levels = np.array(["low", "mid", "high"])
    income_probs = np.array([0.25, 0.55, 0.20])
    income_proxy = rng.choice(income_levels, size=n, p=income_probs)
    channels = np.array(["meta", "organic", "search"])
    channel_probs = np.array([0.35, 0.35, 0.30])
    acquisition_channel_preference = rng.choice(channels, size=n, p=channel_probs)

    # Upgrade 7: spending propensity — lognormal heterogeneity for basket size
    spending_sigma = float(config.get("distribution_realism", {}).get("spending_propensity_sigma", 0.5))
    spending_propensity = np.clip(rng.lognormal(0, spending_sigma, size=n), 0.2, 20.0)

    trust_score = np.full(n, 0.5)
    disappointment_memory = np.zeros(n)
    satisfaction_memory = np.zeros(n)
    recent_negative_velocity = np.zeros(n)
    discount_dependency = np.zeros(n)
    exposure_count = np.zeros(n, dtype=np.int64)
    expressed_desire_level = np.full(n, 0.1, dtype=float)
    desire_decay_memory = np.zeros(n, dtype=float)
    creative_fatigue_map = [{} for _ in range(n)]
    cart_memory = [None] * n
    last_attributed = [None] * n
    shipping_bad_count = np.zeros(n, dtype=np.int64)
    shipping_good_count = np.zeros(n, dtype=np.int64)
    discount_only_buyer = np.zeros(n, dtype=bool)
    days_since_last_interaction = np.zeros(n, dtype=np.int64)

    df = pd.DataFrame({
        "id": prospect_ids,
        "created_at": signup_dates,
        "price_sensitivity": price_sensitivity,
        "impulse_level": impulse_level,
        "loyalty_propensity": loyalty_propensity,
        "regret_propensity": regret_propensity,
        "quality_expectation": quality_expectation,
        "income_proxy": income_proxy,
        "acquisition_channel_preference": acquisition_channel_preference,
        "spending_propensity": spending_propensity,
        "trust_score": trust_score,
        "disappointment_memory": disappointment_memory,
        "satisfaction_memory": satisfaction_memory,
        "recent_negative_velocity": recent_negative_velocity,
        "discount_dependency": discount_dependency,
        "exposure_count": exposure_count,
        "expressed_desire_level": expressed_desire_level,
        "desire_decay_memory": desire_decay_memory,
        "creative_fatigue_map": creative_fatigue_map,
        "cart_memory": cart_memory,
        "shipping_bad_count": shipping_bad_count,
        "shipping_good_count": shipping_good_count,
        "discount_only_buyer": discount_only_buyer,
        "days_since_last_interaction": days_since_last_interaction,
        "last_attributed_ad_id": last_attributed,
        "last_attributed_adset_id": last_attributed,
        "last_attributed_campaign_id": last_attributed,
        "last_attributed_creative_id": last_attributed,
        "last_exposure_date": [pd.NaT] * n,
    })
    df["last_order_date"] = pd.NaT
    df["updated_at"] = df["created_at"]
    df["state"] = "enabled"
    df["verified_email"] = True
    df["email"] = "prospect_" + pd.Series(prospect_ids).str.replace("prospect_", "").str[:12] + "@example.com"
    df["first_name"] = rng.choice(FIRST_NAMES, size=n).tolist()
    df["last_name"] = rng.choice(LAST_NAMES, size=n).tolist()
    df["accepts_marketing"] = rng.random(size=n) < 0.70
    df["default_address_country"] = rng.choice(["US", "CA", "UK", "DE"], size=n).tolist()
    df["default_address_region"] = rng.choice(FAKE_REGIONS, size=n).tolist()
    df["default_address_city"] = rng.choice(FAKE_CITIES, size=n).tolist()
    df["last_order_at"] = pd.NaT
    df["tax_exempt"] = pd.array([pd.NA] * n, dtype="boolean")
    for col in ["phone", "locale", "tags", "note", "source_name", "display_name", "default_address_id", "default_address_postal_code"]:
        df[col] = None
    df["customer_id"] = df["id"].values
    df["signup_date"] = df["created_at"].values
    df["prospect_id"] = df["id"].values
    return df


def get_empty_customers_schema(config: dict) -> pd.DataFrame:
    """Return a DataFrame with 0 rows and the same columns as generate_customers (for first-purchase-creates-customer)."""
    one = generate_customers({**config, "num_customers": 1})
    return one.iloc[0:0].copy()
