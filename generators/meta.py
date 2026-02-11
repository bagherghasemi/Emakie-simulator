"""
Meta-like marketing structure generators for behavioral simulation.

Creatives are psychological recruitment devices: their promise axes (discount,
premium, urgency, safety) will later interact with human latent traits in the
simulator (e.g. price_sensitivity, impulse_level). No BigQuery code; all
functions return pandas DataFrames.
"""

import uuid
from typing import List

import numpy as np
import pandas as pd

PROMISE_AXES = [
    "promise_discount_focus",
    "promise_premium_focus",
    "promise_urgency_focus",
    "promise_safety_focus",
]
OBJECTIVES = ["sales", "traffic"]
AUDIENCE_HINTS = ["broad", "lookalike", "retargeting"]


def generate_creatives(config: dict) -> pd.DataFrame:
    """
    Generate a set of creatives with psychological promise axes.

    Creatives are psychological recruitment devices. Their promise dimensions
    (discount, premium, urgency, safety) are not mutually exclusive and will
    later interact with human latent traits (e.g. price_sensitivity,
    impulse_level, quality_expectation) when simulating ad exposure and
    conversion.

    Count is taken from config (e.g. config["creatives"]). Each creative has
    four promise axes in [0, 1], drawn from a normal distribution around 0.5
    and clamped. A subset of creatives receive mild specialization: one axis
    is boosted so some creatives clearly lean high on that dimension.

    Parameters
    ----------
    config : dict
        Must contain number of creatives (e.g. creatives or num_creatives).
        Optional: seed (int) for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per creative with columns: creative_id, creative_name,
        promise_discount_focus, promise_premium_focus, promise_urgency_focus,
        promise_safety_focus.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    n = config.get("creatives") or config.get("num_creatives")
    if n is None:
        raise ValueError("config must specify number of creatives (e.g. creatives)")
    n = int(n)

    creative_ids = [str(uuid.uuid4()) for _ in range(n)]
    creative_names = [f"Creative {i + 1}" for i in range(n)]

    # Base values: normal around 0.5, clamped to [0, 1]
    promise = np.clip(rng.normal(0.5, 0.2, size=(n, 4)), 0.0, 1.0)

    # Mild specialization: ~25% of creatives get a clear lean on one axis
    n_specialized = max(1, int(n * 0.25))
    specialized_indices = rng.choice(n, size=min(n_specialized, n), replace=False)
    for idx in specialized_indices:
        axis = rng.integers(0, 4)
        # Boost this axis toward 0.75–0.95
        boost = rng.uniform(0.25, 0.45)
        promise[idx, axis] = np.clip(promise[idx, axis] + boost, 0.0, 1.0)

    return pd.DataFrame({
        "creative_id": creative_ids,
        "creative_name": creative_names,
        "promise_discount_focus": promise[:, 0],
        "promise_premium_focus": promise[:, 1],
        "promise_urgency_focus": promise[:, 2],
        "promise_safety_focus": promise[:, 3],
    })


def generate_campaigns(creatives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group creatives into fewer campaigns; one campaign owns many creatives.

    Campaigns are the top-level marketing container. Each has an objective
    (sales or traffic). The number of campaigns is derived so that creatives
    are grouped into fewer entities (e.g. roughly sqrt(n_creatives) to
    n_creatives / 2, with at least 1 campaign).

    Parameters
    ----------
    creatives_df : pandas.DataFrame
        Output of generate_creatives. Must have column: creative_id.

    Returns
    -------
    pandas.DataFrame
        One row per campaign with columns: campaign_id, campaign_name,
        objective.
    """
    seed = getattr(creatives_df, "_generator_seed", None)
    rng = np.random.default_rng(seed)

    n_creatives = len(creatives_df)
    # Fewer campaigns: between 1 and roughly half of creatives, at least 1
    n_campaigns = max(1, rng.integers(1, max(2, n_creatives // 2 + 1)))

    campaign_ids = [str(uuid.uuid4()) for _ in range(n_campaigns)]
    campaign_names = [f"Campaign {i + 1}" for i in range(n_campaigns)]
    objectives = rng.choice(OBJECTIVES, size=n_campaigns)

    return pd.DataFrame({
        "campaign_id": campaign_ids,
        "campaign_name": campaign_names,
        "objective": objectives,
    })


def generate_adsets(campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate multiple adsets per campaign (Meta-like structure).

    Each campaign has several adsets. Each adset has an audience_hint:
    broad, lookalike, or retargeting. The number of adsets per campaign
    is random (e.g. 2–5 per campaign).

    Parameters
    ----------
    campaigns_df : pandas.DataFrame
        Output of generate_campaigns. Must have column: campaign_id.

    Returns
    -------
    pandas.DataFrame
        One row per adset with columns: adset_id, campaign_id, audience_hint.
    """
    seed = getattr(campaigns_df, "_generator_seed", None)
    rng = np.random.default_rng(seed)

    rows: List[dict] = []
    for _, row in campaigns_df.iterrows():
        campaign_id = row["campaign_id"]
        n_adsets = rng.integers(2, 6)  # 2–5 adsets per campaign
        for _ in range(n_adsets):
            rows.append({
                "adset_id": str(uuid.uuid4()),
                "campaign_id": campaign_id,
                "audience_hint": rng.choice(AUDIENCE_HINTS),
            })

    return pd.DataFrame(rows)


def generate_ads(adsets_df: pd.DataFrame, creatives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign one creative per adset; each ad links an adset to a creative.

    Each adset gets exactly one ad, which references one creative from the
    creatives pool. Creatives can be reused across ads (different adsets
    may show the same creative). This produces the leaf-level ad entity
    in the Meta-like hierarchy: campaign → adset → ad → creative.

    Parameters
    ----------
    adsets_df : pandas.DataFrame
        Output of generate_adsets. Must have columns: adset_id.
    creatives_df : pandas.DataFrame
        Output of generate_creatives. Must have column: creative_id.

    Each ad has a delivery_weight (float): normal distribution centered at 1.0,
    std 0.3, clamped to [0.2, 3.0]. It represents how aggressively the platform
    delivers this ad (e.g. higher weight → more delivery relative to others).

    Returns
    -------
    pandas.DataFrame
        One row per ad with columns: ad_id, adset_id, creative_id,
        delivery_weight.
    """
    seed = getattr(adsets_df, "_generator_seed", None)
    rng = np.random.default_rng(seed)

    creative_ids = creatives_df["creative_id"].tolist()
    n_creatives = len(creative_ids)

    rows: List[dict] = []
    for _, row in adsets_df.iterrows():
        adset_id = row["adset_id"]
        creative_id = creative_ids[rng.integers(0, n_creatives)]
        delivery_weight = float(np.clip(rng.normal(1.0, 0.3), 0.2, 3.0))
        rows.append({
            "ad_id": str(uuid.uuid4()),
            "adset_id": adset_id,
            "creative_id": creative_id,
            "delivery_weight": delivery_weight,
        })

    return pd.DataFrame(rows)


def simulate_daily_exposure(
    customers_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    creatives_df: pd.DataFrame,
    current_date: pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    """
    Simulate which customers see and click ads on a given day.

    This function is the recruitment gateway into the system: it determines
    which eligible customers are exposed to which ads and who clicks, based
    on a match between human latent traits and creative promises. Downstream
    simulation (e.g. site visits, conversions) will consume these exposure and
    click events.

    Mechanics:
    - Only customers with signup_date <= current_date are eligible.
    - A random subset of eligible customers (~20–40%) is "active" for the day.
    - Each active customer is shown 1–5 ads, sampled with replacement from
      ads_df with weights proportional to delivery_weight.
    - For each exposure, click probability is computed from trait–promise
      alignment (e.g. high price_sensitivity × promise_discount_focus),
      then base CTR from config, then clamped to [0, 1] and perturbed by
      randomness. Click is drawn from a Bernoulli.

    Parameters
    ----------
    customers_df : pandas.DataFrame
        Must have: customer_id, signup_date, price_sensitivity, impulse_level,
        quality_expectation, regret_propensity.
    ads_df : pandas.DataFrame
        Must have: ad_id, creative_id, delivery_weight.
    creatives_df : pandas.DataFrame
        Must have: creative_id, promise_discount_focus, promise_premium_focus,
        promise_urgency_focus, promise_safety_focus.
    current_date : pandas.Timestamp or date-like
        The simulated day.
    config : dict
        Optional: base_ctr or ctr (float, e.g. 0.01–0.03). Optional: seed (int).

    Returns
    -------
    pandas.DataFrame
        One row per exposure with columns: date, customer_id, ad_id,
        creative_id, impression, clicked.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    current_date = pd.to_datetime(current_date).normalize()
    base_ctr = config.get("base_ctr") or config.get("ctr", 0.02)
    base_ctr = float(base_ctr)

    # Eligible: signup_date <= current_date
    customers_df = customers_df.copy()
    customers_df["signup_date"] = pd.to_datetime(customers_df["signup_date"]).dt.normalize()
    eligible = customers_df[customers_df["signup_date"] <= current_date]
    if eligible.empty:
        return pd.DataFrame(
            columns=["date", "customer_id", "ad_id", "creative_id", "impression", "clicked"]
        )

    # Active subset: ~20–40%
    active_frac = rng.uniform(0.20, 0.40)
    n_eligible = len(eligible)
    n_active = max(1, int(round(n_eligible * active_frac)))
    active_idx = rng.choice(eligible.index, size=min(n_active, n_eligible), replace=False)
    active_customers = eligible.loc[active_idx]
    active_customer_ids = active_customers["customer_id"].tolist()

    # Ad sampling: weights = delivery_weight
    ads = ads_df[["ad_id", "creative_id", "delivery_weight"]].copy()
    weights = ads["delivery_weight"].values.astype(float)
    weights = weights / weights.sum()
    n_ads = len(ads)
    ad_ids_arr = ads["ad_id"].values

    # Build exposures: per active customer, 1–5 ads (weighted sample)
    rows: List[dict] = []
    for cid in active_customer_ids:
        n_exposures = rng.integers(1, 6)  # 1–5 inclusive
        ad_indices = rng.choice(n_ads, size=n_exposures, replace=True, p=weights)
        for idx in ad_indices:
            rows.append({
                "customer_id": cid,
                "ad_id": ad_ids_arr[idx],
            })

    if not rows:
        return pd.DataFrame(
            columns=["date", "customer_id", "ad_id", "creative_id", "impression", "clicked"]
        )

    exposures = pd.DataFrame(rows)
    exposures["date"] = current_date
    exposures["impression"] = 1

    # Merge ad → creative for creative_id (and later promise columns)
    exposures = exposures.merge(ads[["ad_id", "creative_id"]], on="ad_id", how="left")
    # Merge creatives for promise axes
    exposures = exposures.merge(
        creatives_df[
            [
                "creative_id",
                "promise_discount_focus",
                "promise_premium_focus",
                "promise_urgency_focus",
                "promise_safety_focus",
            ]
        ],
        on="creative_id",
        how="left",
    )
    # Merge customer traits
    trait_cols = [
        "customer_id",
        "price_sensitivity",
        "impulse_level",
        "quality_expectation",
        "regret_propensity",
    ]
    exposures = exposures.merge(
        active_customers[trait_cols].drop_duplicates("customer_id"),
        on="customer_id",
        how="left",
    )

    # Vectorized click probability: trait–promise matches nudge probability, not dominate.
    # Base CTR carries most of the level; matches add a small lift for alignment.
    # high price_sensitivity ↔ promise_discount_focus
    # high impulse_level ↔ promise_urgency_focus
    # high quality_expectation ↔ promise_premium_focus
    # low regret_propensity ↔ promise_safety_focus
    match_discount = (
        exposures["price_sensitivity"].values * exposures["promise_discount_focus"].values
    )
    match_urgency = (
        exposures["impulse_level"].values * exposures["promise_urgency_focus"].values
    )
    match_premium = (
        exposures["quality_expectation"].values * exposures["promise_premium_focus"].values
    )
    match_safety = (
        (1.0 - exposures["regret_propensity"].values) * exposures["promise_safety_focus"].values
    )
    # Diminishing returns: average over axes (4) so no single axis dominates.
    # Scale keeps match influence small for realistic paid-social CTR (target 2–5%).
    n_axes = 4
    scale = 0.02  # small nudge from alignment
    noise = rng.normal(0, 0.0025, size=len(exposures))
    match_avg = (match_discount + match_urgency + match_premium + match_safety) / n_axes
    click_prob = base_ctr + scale * match_avg + noise
    click_prob = np.clip(click_prob, 0.0, 1.0)
    clicked = (rng.random(size=len(exposures)) < click_prob).astype(int)

    exposures["clicked"] = clicked

    out = exposures[["date", "customer_id", "ad_id", "creative_id", "impression", "clicked"]]
    return out.copy()