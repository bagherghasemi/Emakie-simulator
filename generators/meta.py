"""
Meta-like marketing structure generators for behavioral simulation.

Creatives are psychological recruitment devices: their promise axes (discount,
premium, urgency, safety) will later interact with human latent traits in the
simulator (e.g. price_sensitivity, impulse_level). No BigQuery code; all
functions return pandas DataFrames.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .psychological_state import _promise_pressure_from_creative


def weighted_budget_split(
    total_budget: float,
    n_children: int,
    concentration: float = 2.0,
    rng: np.random.Generator | None = None,
) -> List[float]:
    """
    Split total_budget across n_children with uneven weights (Dirichlet).
    Sum of returned list equals total_budget (rounding fixed on last item).

    Parameters
    ----------
    total_budget : float
        Total amount to split.
    n_children : int
        Number of shares.
    concentration : float
        Dirichlet alpha; higher = more even split.
    rng : np.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    list[float]
        Non-negative amounts summing to total_budget.
    """
    if n_children <= 0:
        return []
    if n_children == 1:
        return [float(total_budget)]
    rng = rng or np.random.default_rng()
    alpha = np.full(n_children, concentration)
    weights = rng.dirichlet(alpha)
    shares = (weights * total_budget).astype(float)
    # Fix rounding so sum == total_budget (assign residual to last)
    shares[-1] = total_budget - float(np.sum(shares[:-1]))
    shares = np.maximum(shares, 0.0)
    return shares.tolist()

PROMISE_AXES = [
    "promise_discount_focus",
    "promise_premium_focus",
    "promise_urgency_focus",
    "promise_safety_focus",
]
# Canonical 8 psychological promise axes (dbt / Module 3); used for click, expectation, drift when present
PROMISE_8_AXES = [
    "promise_status_intensity",
    "promise_safety_intensity",
    "promise_control_intensity",
    "promise_belonging_intensity",
    "promise_transformation_intensity",
    "promise_relief_intensity",
    "promise_novelty_intensity",
    "promise_mastery_intensity",
]
OBJECTIVES = ["sales", "traffic"]

# Tier 10: audience types with realistic distribution.
# Real Meta accounts are broad-heavy with targeted retargeting/lookalike segments.
AUDIENCE_TYPES = ["broad", "lookalike", "retargeting"]
AUDIENCE_TYPE_PROBS = [0.50, 0.30, 0.20]

# ---------------------------------------------------------------------------
# Tier 9: Creative Cluster Architecture
# ---------------------------------------------------------------------------
# Four causal layers that model real creative strategy structure.
# These are NOT cosmetic metadata — they probabilistically influence promise
# intensities, click probability, fatigue, trust, experience coherence, and
# identity drift.  Outcomes must emerge; nothing is labeled good or bad.
#
# Hierarchy:  value_proposition → creative_angle → hook_pattern → persona
#             ↓ influences promise intensities (8-axis)
#             ↓ propagates through ads (via creative_id join, no duplication)
# ---------------------------------------------------------------------------

# 1) VALUE PROPOSITIONS — deepest layer: what psychological need is activated.
#    Each proposition biases the 8-axis promise intensities (probabilistic, not
#    deterministic).  Format: {axis_name: (mean_shift, std)} applied additively
#    before clamp.  Shifts are modest so randomness still dominates.
VALUE_PROPOSITIONS = [
    "self_improvement",
    "security_and_peace",
    "status_and_recognition",
    "belonging_and_connection",
    "freedom_and_control",
    "discovery_and_novelty",
    "relief_and_comfort",
    "mastery_and_craft",
]

_VP_PROMISE_BIAS: dict[str, dict[str, tuple[float, float]]] = {
    "self_improvement": {
        "promise_transformation_intensity": (0.18, 0.06),
        "promise_mastery_intensity": (0.10, 0.05),
        "promise_control_intensity": (0.06, 0.04),
    },
    "security_and_peace": {
        "promise_safety_intensity": (0.20, 0.06),
        "promise_control_intensity": (0.12, 0.05),
        "promise_relief_intensity": (0.08, 0.04),
    },
    "status_and_recognition": {
        "promise_status_intensity": (0.22, 0.06),
        "promise_transformation_intensity": (0.08, 0.04),
        "promise_belonging_intensity": (0.06, 0.04),
    },
    "belonging_and_connection": {
        "promise_belonging_intensity": (0.20, 0.06),
        "promise_safety_intensity": (0.08, 0.04),
        "promise_relief_intensity": (0.06, 0.04),
    },
    "freedom_and_control": {
        "promise_control_intensity": (0.18, 0.06),
        "promise_novelty_intensity": (0.10, 0.05),
        "promise_status_intensity": (0.06, 0.04),
    },
    "discovery_and_novelty": {
        "promise_novelty_intensity": (0.22, 0.06),
        "promise_transformation_intensity": (0.08, 0.04),
        "promise_mastery_intensity": (0.06, 0.04),
    },
    "relief_and_comfort": {
        "promise_relief_intensity": (0.20, 0.06),
        "promise_safety_intensity": (0.10, 0.05),
        "promise_belonging_intensity": (0.06, 0.04),
    },
    "mastery_and_craft": {
        "promise_mastery_intensity": (0.22, 0.06),
        "promise_control_intensity": (0.10, 0.05),
        "promise_transformation_intensity": (0.08, 0.04),
    },
}

# 2) CREATIVE ANGLES — strategic framing that shapes emotional tone.
#    Behavioral influence parameters per angle (probabilistic modifiers used
#    during simulation, NOT labels):
#      emotional_weight   – scales 8-axis click alignment contribution
#      trust_stability    – modifies post-purchase trust update rate
#      expectation_lift   – shifts expectation score before mismatch calc
#      fatigue_rate       – multiplier on creative fatigue accumulation
#      ctr_volatility     – noise scale on click probability
CREATIVE_ANGLES = [
    "problem_solution",
    "transformation",
    "authority",
    "social_proof",
    "fear_based",
    "educational",
    "lifestyle",
]

_ANGLE_PARAMS: dict[str, dict[str, float]] = {
    "problem_solution":  {"emotional_weight": 1.00, "trust_stability": 1.05, "expectation_lift": 0.05, "fatigue_rate": 1.00, "ctr_volatility": 1.00},
    "transformation":    {"emotional_weight": 1.15, "trust_stability": 0.95, "expectation_lift": 0.12, "fatigue_rate": 1.05, "ctr_volatility": 1.05},
    "authority":         {"emotional_weight": 0.90, "trust_stability": 1.20, "expectation_lift": 0.03, "fatigue_rate": 0.90, "ctr_volatility": 0.85},
    "social_proof":      {"emotional_weight": 1.05, "trust_stability": 1.10, "expectation_lift": 0.06, "fatigue_rate": 0.95, "ctr_volatility": 0.95},
    "fear_based":        {"emotional_weight": 1.25, "trust_stability": 0.80, "expectation_lift": 0.15, "fatigue_rate": 1.15, "ctr_volatility": 1.25},
    "educational":       {"emotional_weight": 0.85, "trust_stability": 1.15, "expectation_lift": 0.02, "fatigue_rate": 0.85, "ctr_volatility": 0.80},
    "lifestyle":         {"emotional_weight": 1.10, "trust_stability": 1.00, "expectation_lift": 0.08, "fatigue_rate": 1.00, "ctr_volatility": 1.10},
}

# 3) HOOK PATTERNS — first-3-second tactical surface layer.
#    Hooks influence initial attention (CTR variance, bounce) and short-term
#    fatigue.  They do NOT affect long-term trust directly.
#      attention_spike  – additive CTR boost (short-lived)
#      bounce_risk      – additive penalty on landing page view probability
#      fatigue_accel    – multiplier on hook-specific fatigue accumulation
#      novelty_decay    – how fast the hook loses surprise (fatigue sensitivity)
HOOK_PATTERNS = [
    "question",
    "bold_claim",
    "story_open",
    "pattern_interrupt",
    "testimonial_lead",
    "statistic",
    "curiosity_gap",
    "before_after",
]

_HOOK_PARAMS: dict[str, dict[str, float]] = {
    "question":          {"attention_spike": 0.004, "bounce_risk": 0.00, "fatigue_accel": 1.00, "novelty_decay": 1.00},
    "bold_claim":        {"attention_spike": 0.008, "bounce_risk": 0.02, "fatigue_accel": 1.20, "novelty_decay": 1.30},
    "story_open":        {"attention_spike": 0.003, "bounce_risk":-0.01, "fatigue_accel": 0.85, "novelty_decay": 0.80},
    "pattern_interrupt": {"attention_spike": 0.010, "bounce_risk": 0.03, "fatigue_accel": 1.35, "novelty_decay": 1.50},
    "testimonial_lead":  {"attention_spike": 0.005, "bounce_risk":-0.01, "fatigue_accel": 0.90, "novelty_decay": 0.90},
    "statistic":         {"attention_spike": 0.006, "bounce_risk": 0.01, "fatigue_accel": 1.05, "novelty_decay": 1.15},
    "curiosity_gap":     {"attention_spike": 0.009, "bounce_risk": 0.02, "fatigue_accel": 1.25, "novelty_decay": 1.40},
    "before_after":      {"attention_spike": 0.007, "bounce_risk": 0.00, "fatigue_accel": 1.10, "novelty_decay": 1.10},
}

# 4) PERSONAS — messaging point of view.
#    Personas probabilistically attract certain trait distributions.
#    Format: {trait: (affinity_center, affinity_width)} — Gaussian affinity;
#    click probability is modified by how close human traits are to the
#    persona's resonance center.  Does NOT restrict who sees the ad.
CREATIVE_PERSONAS = [
    "busy_parents",
    "ambitious_professionals",
    "health_conscious",
    "trend_seekers",
    "value_hunters",
    "comfort_lovers",
    "creative_makers",
    "community_builders",
]

_PERSONA_TRAIT_AFFINITY: dict[str, dict[str, tuple[float, float]]] = {
    "busy_parents": {
        "loyalty_propensity": (0.70, 0.20),
        "impulse_level": (0.55, 0.20),
        "regret_propensity": (0.45, 0.15),
    },
    "ambitious_professionals": {
        "quality_expectation": (0.75, 0.18),
        "impulse_level": (0.40, 0.18),
        "loyalty_propensity": (0.50, 0.20),
    },
    "health_conscious": {
        "quality_expectation": (0.70, 0.18),
        "regret_propensity": (0.55, 0.18),
        "price_sensitivity": (0.35, 0.18),
    },
    "trend_seekers": {
        "impulse_level": (0.72, 0.18),
        "price_sensitivity": (0.40, 0.20),
        "quality_expectation": (0.55, 0.20),
    },
    "value_hunters": {
        "price_sensitivity": (0.78, 0.16),
        "impulse_level": (0.60, 0.18),
        "loyalty_propensity": (0.35, 0.20),
    },
    "comfort_lovers": {
        "regret_propensity": (0.60, 0.18),
        "loyalty_propensity": (0.65, 0.18),
        "impulse_level": (0.35, 0.20),
    },
    "creative_makers": {
        "quality_expectation": (0.65, 0.20),
        "loyalty_propensity": (0.60, 0.18),
        "impulse_level": (0.50, 0.20),
    },
    "community_builders": {
        "loyalty_propensity": (0.75, 0.16),
        "regret_propensity": (0.40, 0.18),
        "quality_expectation": (0.55, 0.20),
    },
}

# 5) UPGRADED CREATIVE TYPE — richer taxonomy: format × production style × voice.
#    Influences watch time, fatigue curve, trust baseline, novelty decay.
#    Does NOT override psychological alignment.
CREATIVE_TYPES = [
    "static_brand_polished",
    "static_ugc_raw",
    "video_brand_polished",
    "video_ugc_raw",
    "video_creator_collab",
    "carousel_editorial",
    "carousel_ugc",
    "dynamic_product",
    "story_native",
]
CREATIVE_TYPE_PROBS = [0.12, 0.10, 0.14, 0.13, 0.10, 0.10, 0.08, 0.13, 0.10]

_CREATIVE_TYPE_PARAMS: dict[str, dict[str, float]] = {
    "static_brand_polished":  {"watch_time_mult": 0.0, "fatigue_curve": 0.95, "trust_baseline_shift": 0.03, "novelty_decay": 1.00},
    "static_ugc_raw":         {"watch_time_mult": 0.0, "fatigue_curve": 0.85, "trust_baseline_shift": 0.05, "novelty_decay": 0.85},
    "video_brand_polished":   {"watch_time_mult": 1.15, "fatigue_curve": 1.00, "trust_baseline_shift": 0.04, "novelty_decay": 1.05},
    "video_ugc_raw":          {"watch_time_mult": 1.05, "fatigue_curve": 0.80, "trust_baseline_shift": 0.06, "novelty_decay": 0.75},
    "video_creator_collab":   {"watch_time_mult": 1.20, "fatigue_curve": 0.85, "trust_baseline_shift": 0.07, "novelty_decay": 0.80},
    "carousel_editorial":     {"watch_time_mult": 0.0, "fatigue_curve": 1.05, "trust_baseline_shift": 0.02, "novelty_decay": 1.10},
    "carousel_ugc":           {"watch_time_mult": 0.0, "fatigue_curve": 0.90, "trust_baseline_shift": 0.05, "novelty_decay": 0.90},
    "dynamic_product":        {"watch_time_mult": 0.0, "fatigue_curve": 1.10, "trust_baseline_shift": 0.01, "novelty_decay": 1.20},
    "story_native":           {"watch_time_mult": 1.10, "fatigue_curve": 0.75, "trust_baseline_shift": 0.06, "novelty_decay": 0.70},
}


def _is_video_type(creative_type: str) -> bool:
    """Whether creative_type produces video content (for watch time / thruplay)."""
    return creative_type in ("video_brand_polished", "video_ugc_raw", "video_creator_collab", "story_native")


def _persona_resonance(
    persona: str,
    trait_values: dict[str, float],
) -> float:
    """Compute [0, 1] resonance between a persona and a human's trait values.

    Uses Gaussian affinity: for each trait the persona cares about, measure
    how close the human's trait value is to the persona's center.  Average
    across traits, then scale to [0, 1].  Higher = stronger resonance.
    """
    affinity = _PERSONA_TRAIT_AFFINITY.get(persona)
    if not affinity:
        return 0.5
    scores = []
    for trait, (center, width) in affinity.items():
        val = trait_values.get(trait, 0.5)
        dist = abs(val - center)
        score = np.exp(-0.5 * (dist / max(width, 0.01)) ** 2)
        scores.append(float(score))
    return float(np.mean(scores)) if scores else 0.5


def _deterministic_uuid4(rng: np.random.Generator) -> str:
    """Generate a deterministic UUID4 from an RNG (for reproducible runs)."""
    b = bytearray(rng.integers(0, 256, size=16, dtype=np.uint8).tolist())
    b[6] = (b[6] & 0x0F) | 0x40
    b[8] = (b[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(b)))


def generate_ad_accounts(config: dict) -> pd.DataFrame:
    """
    Generate Meta ad accounts (top-level entity in the hierarchy).

    Output matches dbt meta_ad_accounts source: id, name, currency,
    timezone_name; NULL for account_status, created_time, etc.
    ad_account_id is present as alias of id for downstream (e.g. generate_campaigns).

    Parameters
    ----------
    config : dict
        ad_accounts (int): number of accounts (default 1).
        seed (int, optional): for deterministic generation.

    Returns
    -------
    pandas.DataFrame
        Columns: id, name, currency, timezone_name; NULL: account_status,
        is_personal, is_prepay_account, timezone_offset_hours_utc, amount_spent,
        balance, spend_cap, created_time, updated_time, business_id, business_name,
        owner, admin_graphql_api_id. ad_account_id = id.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    n = config.get("ad_accounts", 1)
    n = int(n)

    ids = [_deterministic_uuid4(rng) for _ in range(n)]
    names = [f"Ad Account {i + 1}" for i in range(n)]
    currency = "USD"
    timezone_name = "UTC"

    df = pd.DataFrame({
        "id": ids,
        "name": names,
        "currency": [currency] * n,
        "timezone_name": [timezone_name] * n,
    })

    # Typed nulls per contract: TIMESTAMP → pd.NaT, NUMERIC → np.nan, BOOL → nullable bool, STRING → None
    df["created_time"] = pd.NaT
    df["updated_time"] = pd.NaT
    df["timezone_offset_hours_utc"] = np.nan
    df["amount_spent"] = np.nan
    df["balance"] = np.nan
    df["spend_cap"] = np.nan
    df["is_personal"] = pd.array([pd.NA] * len(df), dtype="boolean")
    df["is_prepay_account"] = pd.array([pd.NA] * len(df), dtype="boolean")
    for col in ["account_status", "business_id", "business_name", "owner", "admin_graphql_api_id"]:
        df[col] = None

    # Alias for downstream (generate_campaigns)
    df["ad_account_id"] = df["id"].values

    return df


def generate_creatives(config: dict) -> pd.DataFrame:
    """
    Generate creatives with four causal creative-strategy layers plus
    psychological promise axes.

    Tier 9 hierarchy per creative:
        value_proposition  → deepest: what psychological need is activated
        creative_angle     → strategic framing / emotional tone
        hook_pattern       → first-3-second tactical surface
        persona            → messaging point-of-view / audience affinity
        creative_type      → format × production style × voice
        promise intensities → 8-axis (influenced by value_proposition)

    Value proposition probabilistically biases promise intensities so that
    creatives with the same proposition share a recognizable promise
    fingerprint while retaining variance.

    Parameters
    ----------
    config : dict
        Must contain number of creatives (e.g. creatives or num_creatives).
        Optional: seed (int) for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per creative.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    n = config.get("creatives") or config.get("num_creatives")
    if n is None:
        raise ValueError("config must specify number of creatives (e.g. creatives)")
    n = int(n)

    creative_ids = [str(uuid.uuid4()) for _ in range(n)]
    creative_names = [f"Creative {i + 1}" for i in range(n)]

    # --- Tier 9: four creative-strategy layers ---
    value_propositions = rng.choice(VALUE_PROPOSITIONS, size=n, replace=True).tolist()
    angles = rng.choice(CREATIVE_ANGLES, size=n, replace=True).tolist()
    hooks = rng.choice(HOOK_PATTERNS, size=n, replace=True).tolist()
    personas = rng.choice(CREATIVE_PERSONAS, size=n, replace=True).tolist()

    creative_types = rng.choice(
        CREATIVE_TYPES, size=n, replace=True, p=CREATIVE_TYPE_PROBS
    ).tolist()

    # --- Legacy 4 promise axes (base normal around 0.5) ---
    promise = np.clip(rng.normal(0.5, 0.2, size=(n, 4)), 0.0, 1.0)
    n_specialized = max(1, int(n * 0.25))
    specialized_indices = rng.choice(n, size=min(n_specialized, n), replace=False)
    for idx in specialized_indices:
        axis = rng.integers(0, 4)
        boost = rng.uniform(0.25, 0.45)
        promise[idx, axis] = np.clip(promise[idx, axis] + boost, 0.0, 1.0)

    # --- Canonical 8 promise axes (base normal, then value_proposition bias) ---
    promise_8_names = [
        "promise_status_intensity",
        "promise_safety_intensity",
        "promise_control_intensity",
        "promise_belonging_intensity",
        "promise_transformation_intensity",
        "promise_relief_intensity",
        "promise_novelty_intensity",
        "promise_mastery_intensity",
    ]
    promise_8 = np.clip(rng.normal(0.5, 0.2, size=(n, 8)), 0.0, 1.0)
    promise_8[:, 1] = promise[:, 3]  # safety mapped from legacy

    # Mild random specialization (before VP bias)
    n_spec_8 = max(1, int(n * 0.25))
    spec_8_idx = rng.choice(n, size=min(n_spec_8, n), replace=False)
    for idx in spec_8_idx:
        axis = rng.integers(0, 8)
        if axis == 1:
            continue
        boost = rng.uniform(0.25, 0.45)
        promise_8[idx, axis] = np.clip(promise_8[idx, axis] + boost, 0.0, 1.0)

    # Value proposition → probabilistic promise bias (stable fingerprint per VP)
    p8_name_to_col = {name: j for j, name in enumerate(promise_8_names)}
    for i in range(n):
        vp = value_propositions[i]
        biases = _VP_PROMISE_BIAS.get(vp, {})
        for axis_name, (mean_shift, std) in biases.items():
            col = p8_name_to_col.get(axis_name)
            if col is not None:
                nudge = rng.normal(mean_shift, std)
                promise_8[i, col] = np.clip(promise_8[i, col] + nudge, 0.0, 1.0)

    # video_duration_seconds: video types get 15–60s, others 0
    video_duration_seconds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if _is_video_type(creative_types[i]):
            video_duration_seconds[i] = int(rng.integers(15, 61))

    # Upgrade 7: innate creative quality — lognormal heterogeneity (some creatives are just better)
    quality_sigma = float(config.get("distribution_realism", {}).get("creative_quality_sigma", 0.4))
    innate_quality = np.clip(rng.lognormal(0, quality_sigma, size=n), 0.3, 3.0)

    out = pd.DataFrame({
        "creative_id": creative_ids,
        "creative_name": creative_names,
        "creative_type": creative_types,
        "creative_value_proposition": value_propositions,
        "creative_angle": angles,
        "creative_hook_pattern": hooks,
        "creative_persona": personas,
        "video_duration_seconds": video_duration_seconds,
        "innate_quality": innate_quality,
        "promise_discount_focus": promise[:, 0],
        "promise_premium_focus": promise[:, 1],
        "promise_urgency_focus": promise[:, 2],
        "promise_safety_focus": promise[:, 3],
    })
    for j, name in enumerate(promise_8_names):
        out[name] = promise_8[:, j]

    # Upgrade 9: creative lifecycle initial state
    out["total_impressions"] = 0
    out["days_active"] = 0
    out["lifecycle_stage"] = "launch"
    out["performance_multiplier"] = 0.8  # launch starts below peak

    return out


# Budget tier ranges (float): small 500–5k, mid 5k–50k, large 50k–500k
BUDGET_TIER_RANGES = {
    "small": (500.0, 5000.0),
    "mid": (5000.0, 50000.0),
    "large": (50000.0, 500000.0),
}


def generate_campaigns(
    creatives_df: pd.DataFrame,
    ad_accounts_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Group creatives into fewer campaigns; one campaign owns many creatives.

    Campaign owns total budget and flight dates. Output matches dbt meta_campaigns:
    id, name, objective, account_id, start_time, stop_time, lifetime_budget;
    daily_budget optional (may be null). campaign_id = id.

    Parameters
    ----------
    creatives_df : pandas.DataFrame
        Output of generate_creatives. Must have column: creative_id.
    ad_accounts_df : pandas.DataFrame
        Output of generate_ad_accounts. Must have column: ad_account_id.
    config : dict
        simulation.start_date, simulation.end_date; optional budget_tier (small/mid/large)
        or budget_min, budget_max (floats).
    """
    seed = config.get("seed") or getattr(creatives_df, "_generator_seed", None)
    rng = np.random.default_rng(seed)

    sim = config.get("simulation") or {}
    start_str = sim.get("start_date", "2023-01-01")
    end_str = sim.get("end_date", "2026-01-01")
    sim_start = pd.Timestamp(start_str).normalize()
    sim_end = pd.Timestamp(end_str).normalize()
    sim_range_days = (sim_end - sim_start).days + 1

    n_creatives = len(creatives_df)
    n_campaigns = max(1, rng.integers(1, max(2, n_creatives // 2 + 1)))

    ids = [str(uuid.uuid4()) for _ in range(n_campaigns)]
    names = [f"Campaign {i + 1}" for i in range(n_campaigns)]
    objectives = rng.choice(OBJECTIVES, size=n_campaigns)

    account_ids = ad_accounts_df["ad_account_id"].tolist()
    account_id_values = rng.choice(account_ids, size=n_campaigns, replace=True)

    # Budget: scenario distribution (lognormal or uniform in tier range)
    tier = config.get("budget_tier", "mid")
    if tier in BUDGET_TIER_RANGES:
        low, high = BUDGET_TIER_RANGES[tier]
    else:
        low = float(config.get("budget_min", 5000.0))
        high = float(config.get("budget_max", 50000.0))
    lifetime_budgets = rng.uniform(low, high, size=n_campaigns).astype(float)

    # start_time: simulation start OR random in first 20% of range
    first_20pct = max(0, int(sim_range_days * 0.2))
    start_offsets = rng.integers(0, max(1, first_20pct + 1), size=n_campaigns)
    start_times = [sim_start + pd.Timedelta(days=int(o)) for o in start_offsets]

    # stop_time: campaigns run for 60-100% of remaining sim period (ensures ad spend throughout)
    stop_times: List[pd.Timestamp] = []
    for i in range(n_campaigns):
        start = start_times[i]
        remaining = (sim_end - start).days + 1
        remaining = max(1, remaining)
        min_duration = max(1, int(remaining * 0.6))
        offset = rng.integers(min_duration, remaining + 1)
        stop_times.append(start + pd.Timedelta(days=min(offset, remaining)))

    df = pd.DataFrame({
        "id": ids,
        "name": names,
        "objective": objectives,
        "account_id": account_id_values,
        "start_time": start_times,
        "stop_time": stop_times,
        "lifetime_budget": lifetime_budgets,
    })

    # Typed nulls per contract: TIMESTAMP → pd.NaT, NUMERIC → np.nan, STRING → None
    df["daily_budget"] = np.nan
    df["created_time"] = pd.NaT
    df["updated_time"] = pd.NaT
    df["budget_remaining"] = np.nan
    for col in ["buying_type", "special_ad_categories", "status", "effective_status", "configured_status", "source_campaign_id", "promoted_object", "admin_graphql_api_id"]:
        df[col] = None

    df["campaign_id"] = df["id"].values

    return df


def generate_adsets(campaigns_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generate multiple adsets per campaign; each receives a weighted share of campaign budget.

    Tier 10: each adset gets an audience_type (broad / lookalike / retargeting)
    with realistic distribution.  audience_type is a real acquisition dimension
    visible to dbt, not a simulator-internal flag.

    Ad set start_time >= campaign start_time; end_time <= campaign stop_time.
    Output: id, campaign_id, audience_type, lifetime_budget, start_time, end_time.
    adset_id = id.
    """
    seed = config.get("seed") or getattr(campaigns_df, "_generator_seed", None)
    rng = np.random.default_rng(seed)
    concentration = float(config.get("budget_concentration", 2.0))

    rows: List[dict] = []
    for _, row in campaigns_df.iterrows():
        campaign_id = row["campaign_id"]
        camp_start = pd.Timestamp(row["start_time"]).normalize()
        camp_stop = pd.Timestamp(row["stop_time"]).normalize()
        camp_budget = float(row["lifetime_budget"])
        n_adsets = rng.integers(2, 6)
        adset_budgets = weighted_budget_split(camp_budget, n_adsets, concentration, rng)
        camp_days = max(1, (camp_stop - camp_start).days + 1)

        adset_audience_types = rng.choice(
            AUDIENCE_TYPES, size=n_adsets, replace=True, p=AUDIENCE_TYPE_PROBS,
        ).tolist()

        for k in range(n_adsets):
            offset_start = rng.integers(0, max(1, camp_days // 3))
            offset_end = rng.integers(offset_start, camp_days)
            start_time = camp_start + pd.Timedelta(days=offset_start)
            end_time = camp_start + pd.Timedelta(days=offset_end)
            rows.append({
                "id": str(uuid.uuid4()),
                "campaign_id": campaign_id,
                "audience_type": adset_audience_types[k],
                "lifetime_budget": adset_budgets[k],
                "start_time": start_time,
                "end_time": end_time,
            })

    df = pd.DataFrame(rows)

    df["daily_budget"] = np.nan
    df["budget_remaining"] = np.nan
    df["bid_amount"] = np.nan
    df["created_time"] = pd.NaT
    df["updated_time"] = pd.NaT
    for col in ["account_id", "name", "status", "effective_status", "configured_status", "billing_event", "optimization_goal", "targeting", "promoted_object", "source_ad_set_id", "admin_graphql_api_id"]:
        df[col] = None

    df["adset_id"] = df["id"].values

    return df


def generate_ads(adsets_df: pd.DataFrame, creatives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign one creative per adset; each ad links an adset to a creative.

    Each adset gets exactly one ad, which references one creative from the
    creatives pool. Output matches dbt meta_ads source: id, adset_id,
    creative_id; NULL for campaign_id, account_id, name, status, etc.
    delivery_weight is not included (simulator internal). ad_id = id for downstream.

    Parameters
    ----------
    adsets_df : pandas.DataFrame
        Output of generate_adsets. Must have columns: adset_id.
    creatives_df : pandas.DataFrame
        Output of generate_creatives. Must have column: creative_id.

    Returns
    -------
    pandas.DataFrame
        One row per ad. Columns: id, adset_id, creative_id; NULL: campaign_id,
        account_id, name, status, effective_status, configured_status,
        creative_name, creative_type, preview_url, created_time, updated_time,
        start_time, end_time, tracking_specs, conversion_domain, bid_amount,
        bid_type, source_ad_id, admin_graphql_api_id. ad_id = id.
    """
    seed = getattr(adsets_df, "_generator_seed", None)
    rng = np.random.default_rng(seed)

    creative_ids = creatives_df["creative_id"].tolist()
    n_creatives = len(creative_ids)

    rows: List[dict] = []
    for _, row in adsets_df.iterrows():
        adset_id = row["adset_id"]
        creative_id = creative_ids[rng.integers(0, n_creatives)]
        # delivery_weight: for spend distribution in performance (no budget on ads)
        dw = float(rng.uniform(0.5, 2.0))
        rows.append({
            "id": str(uuid.uuid4()),
            "adset_id": adset_id,
            "creative_id": creative_id,
            "delivery_weight": dw,
        })

    df = pd.DataFrame(rows)

    # Typed nulls per contract: TIMESTAMP → pd.NaT, NUMERIC → np.nan, STRING → None
    df["created_time"] = pd.NaT
    df["updated_time"] = pd.NaT
    df["start_time"] = pd.NaT
    df["end_time"] = pd.NaT
    df["bid_amount"] = np.nan
    for col in ["campaign_id", "account_id", "name", "status", "effective_status", "configured_status", "creative_name", "creative_type", "preview_url", "tracking_specs", "conversion_domain", "bid_type", "source_ad_id", "admin_graphql_api_id"]:
        df[col] = None

    df["ad_id"] = df["id"].values

    return df


def simulate_daily_exposure(
    customers_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    creatives_df: pd.DataFrame,
    current_date: pd.Timestamp,
    config: dict,
    *,
    exposure_count_map: Optional[Dict[Tuple[str, str], int]] = None,
    last_exposure_by_customer: Optional[Dict[str, Any]] = None,
    brand_state=None,
    seasonality_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Simulate which customers see and click ads on a given day.

    Tier 10: delivery is a probabilistic bias system grounded in behavioral
    state, campaign objective, and audience type.

    Mechanics:
    - Only humans with signup_date <= current_date are eligible.
    - A random active subset (~20-40%) is selected for the day.
    - For each active human, ads are sampled with per-ad delivery weights
      that combine:
        * Base delivery_weight (budget allocation)
        * Audience type affinity (retargeting prefers prior-behavior humans,
          lookalike prefers trait-similar, broad has minimal bias)
        * Objective optimization (sales prefers convert-likely, traffic
          prefers click-likely)
    - Delivery weights are soft/probabilistic, never hard filters.
    - Click probability then depends on trait-promise alignment + Tier 9
      creative layers (unchanged from before).

    Parameters
    ----------
    customers_df : pandas.DataFrame
        Must have: customer_id, signup_date, and latent trait columns.
    ads_df : pandas.DataFrame
        Must have: ad_id, creative_id. Optional: delivery_weight, audience_type, objective.
    creatives_df : pandas.DataFrame
        Must have: creative_id, promise axes.
    current_date : pandas.Timestamp or date-like
        The simulated day.
    config : dict
        Optional: base_ctr, seed, targeting params.

    Returns
    -------
    pandas.DataFrame
        One row per exposure.
    """
    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    current_date = pd.to_datetime(current_date).normalize()
    base_ctr = config.get("base_ctr") or config.get("ctr", 0.02)
    base_ctr = float(base_ctr)

    # Upgrade 1: Brand lifecycle phase effect on CTR
    if brand_state is not None:
        base_ctr *= brand_state.get_phase_effects().get("ctr_mult", 1.0)

    # Eligible: signup_date <= current_date
    customers_df = customers_df.copy()
    customers_df["signup_date"] = pd.to_datetime(customers_df["signup_date"]).dt.normalize()
    eligible = customers_df[customers_df["signup_date"] <= current_date]
    _raw_signal_cols = [
        "repeat_creative_exposure_count", "days_since_last_exposure",
        "multi_ad_session_depth", "engagement_intensity_score",
    ]
    if eligible.empty:
        return pd.DataFrame(
            columns=["date", "customer_id", "ad_id", "creative_id", "impression", "clicked",
                     "watch_time_seconds", "liked", "saved", "shared"] + _raw_signal_cols
        )

    # Active subset: ~20-40%, modulated by seasonality
    active_frac = rng.uniform(0.20, 0.40) * seasonality_mult
    active_frac = min(active_frac, 1.0)  # cap at 100%
    n_eligible = len(eligible)
    n_active = max(1, int(round(n_eligible * active_frac)))
    active_idx = rng.choice(eligible.index, size=min(n_active, n_eligible), replace=False)
    active_customers = eligible.loc[active_idx]
    active_customer_ids = active_customers["customer_id"].tolist()

    # --- Tier 10: behaviorally-biased delivery scoring ---
    # Pre-compute per-human behavioral signals for delivery scoring.
    # These are existing simulator state — no new entities.
    cid_to_row: Dict[str, int] = {
        str(cid): i for i, cid in enumerate(active_customers["customer_id"])
    }
    n_cust = len(active_customers)
    h_exposure_count = active_customers["exposure_count"].fillna(0).values.astype(float) if "exposure_count" in active_customers.columns else np.zeros(n_cust)
    h_trust = active_customers["trust_score"].fillna(0.5).values.astype(float) if "trust_score" in active_customers.columns else np.full(n_cust, 0.5)
    h_disappointment = active_customers["disappointment_memory"].fillna(0).values.astype(float) if "disappointment_memory" in active_customers.columns else np.zeros(n_cust)
    h_impulse = active_customers["impulse_level"].fillna(0.5).values.astype(float)
    h_loyalty = active_customers["loyalty_propensity"].fillna(0.5).values.astype(float) if "loyalty_propensity" in active_customers.columns else np.full(n_cust, 0.5)
    h_desire = active_customers["expressed_desire_level"].fillna(0.1).values.astype(float) if "expressed_desire_level" in active_customers.columns else np.full(n_cust, 0.1)
    h_has_cart = np.zeros(n_cust, dtype=bool)
    if "cart_memory" in active_customers.columns:
        h_has_cart = active_customers["cart_memory"].apply(
            lambda x: x is not None and isinstance(x, dict)
        ).values
    h_has_prior_exposure = h_exposure_count > 0
    h_has_purchase = np.zeros(n_cust, dtype=bool)
    if "last_order_date" in active_customers.columns:
        h_has_purchase = active_customers["last_order_date"].notna().values
    h_satisfaction = active_customers["satisfaction_memory"].fillna(0).values.astype(float) if "satisfaction_memory" in active_customers.columns else np.zeros(n_cust)

    # Pre-compute per-ad base weights + targeting metadata
    ads = ads_df[["ad_id", "creative_id"]].copy()
    n_ads = len(ads)
    ad_ids_arr = ads["ad_id"].values

    base_weights = ads_df["delivery_weight"].values.astype(float) if "delivery_weight" in ads_df.columns else np.ones(n_ads)
    ad_audience_type = ads_df["audience_type"].values if "audience_type" in ads_df.columns else np.array(["broad"] * n_ads)
    ad_objective = ads_df["objective"].values if "objective" in ads_df.columns else np.array(["sales"] * n_ads)

    # Pre-compute per-ad audience type and objective modifier arrays (vectorized lookups)
    ad_is_retargeting = np.array([at == "retargeting" for at in ad_audience_type])
    ad_is_lookalike = np.array([at == "lookalike" for at in ad_audience_type])
    ad_is_broad = np.array([at == "broad" for at in ad_audience_type])
    ad_is_sales = np.array([obj == "sales" for obj in ad_objective])
    ad_is_traffic = np.array([obj == "traffic" for obj in ad_objective])

    # Targeting strength parameters (configurable, probabilistic not deterministic)
    retarget_prior_exposure_boost = float(config.get("retarget_prior_exposure_boost", 3.0))
    retarget_cart_boost = float(config.get("retarget_cart_boost", 5.0))
    retarget_purchase_boost = float(config.get("retarget_purchase_boost", 2.0))
    retarget_desire_boost = float(config.get("retarget_desire_boost", 2.5))
    retarget_base_leak = float(config.get("retarget_base_leak", 0.15))

    lookalike_trust_boost = float(config.get("lookalike_trust_boost", 1.8))
    lookalike_loyalty_boost = float(config.get("lookalike_loyalty_boost", 1.5))
    lookalike_satisfaction_boost = float(config.get("lookalike_satisfaction_boost", 1.5))

    sales_trust_weight = float(config.get("sales_obj_trust_weight", 1.5))
    sales_desire_weight = float(config.get("sales_obj_desire_weight", 1.3))
    sales_loyalty_weight = float(config.get("sales_obj_loyalty_weight", 1.2))
    traffic_impulse_weight = float(config.get("traffic_obj_impulse_weight", 1.5))
    traffic_novelty_weight = float(config.get("traffic_obj_novelty_weight", 1.3))

    # Build exposures: per active customer, 1-5 ads with behaviorally-biased sampling
    rows: List[dict] = []
    for c_idx, cid in enumerate(active_customer_ids):
        i = cid_to_row[str(cid)]
        n_exposures = rng.integers(1, 6)

        # Start from base delivery weights
        w = base_weights.copy()

        # --- Audience type delivery bias (soft, probabilistic) ---

        # Retargeting: boost for humans with prior behavioral signals
        retarget_score = retarget_base_leak
        if h_has_prior_exposure[i]:
            retarget_score += retarget_prior_exposure_boost * min(h_exposure_count[i] / 10.0, 1.0)
        if h_has_cart[i]:
            retarget_score += retarget_cart_boost
        if h_has_purchase[i]:
            retarget_score += retarget_purchase_boost
        if h_desire[i] > 0.3:
            retarget_score += retarget_desire_boost * h_desire[i]
        retarget_score = min(retarget_score, 15.0)
        w[ad_is_retargeting] *= retarget_score

        # Lookalike: boost for humans whose traits resemble high-value customers
        lookalike_score = 0.5 + (
            lookalike_trust_boost * max(0, h_trust[i] - 0.4)
            + lookalike_loyalty_boost * max(0, h_loyalty[i] - 0.4)
            + lookalike_satisfaction_boost * max(0, h_satisfaction[i] - 0.1)
        )
        lookalike_score = min(lookalike_score, 8.0)
        w[ad_is_lookalike] *= lookalike_score

        # Broad: minimal bias, stays close to base weights (slight freshness preference)
        broad_freshness = 1.0 + 0.3 * (1.0 - min(h_exposure_count[i] / 20.0, 1.0))
        w[ad_is_broad] *= broad_freshness

        # --- Objective optimization bias (soft) ---

        # Sales: prefer humans likely to convert
        sales_score = 0.5 + (
            sales_trust_weight * max(0, h_trust[i] - 0.3)
            + sales_desire_weight * h_desire[i]
            + sales_loyalty_weight * max(0, h_loyalty[i] - 0.3)
        )
        sales_score *= (1.0 - 0.5 * min(h_disappointment[i], 1.0))
        if h_has_cart[i]:
            sales_score *= 1.5
        sales_score = min(sales_score, 10.0)
        w[ad_is_sales] *= sales_score

        # Traffic: prefer humans likely to click
        traffic_score = 0.5 + (
            traffic_impulse_weight * h_impulse[i]
            + traffic_novelty_weight * (1.0 - min(h_exposure_count[i] / 15.0, 1.0))
        )
        traffic_score = min(traffic_score, 8.0)
        w[ad_is_traffic] *= traffic_score

        # Add small noise to prevent perfectly deterministic delivery
        w += rng.exponential(0.05, size=n_ads)
        w = np.maximum(w, 1e-8)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones(n_ads) / n_ads

        ad_indices = rng.choice(n_ads, size=n_exposures, replace=True, p=w)
        for idx in ad_indices:
            rows.append({
                "customer_id": cid,
                "ad_id": ad_ids_arr[idx],
            })

    if not rows:
        return pd.DataFrame(
            columns=["date", "customer_id", "ad_id", "creative_id", "impression", "clicked",
                     "watch_time_seconds", "liked", "saved", "shared"] + _raw_signal_cols
        )

    exposures = pd.DataFrame(rows)
    exposures["date"] = current_date
    exposures["impression"] = 1

    # Merge ad → creative for creative_id (and later promise columns + creative_type + Tier 9 layers)
    exposures = exposures.merge(ads[["ad_id", "creative_id"]], on="ad_id", how="left")
    creative_cols = [
        "creative_id",
        "promise_discount_focus",
        "promise_premium_focus",
        "promise_urgency_focus",
        "promise_safety_focus",
    ]
    for ax in PROMISE_8_AXES:
        if ax in creatives_df.columns and ax not in creative_cols:
            creative_cols.append(ax)
    for tier9_col in ["creative_type", "video_duration_seconds",
                       "creative_value_proposition", "creative_angle",
                       "creative_hook_pattern", "creative_persona",
                       "innate_quality", "performance_multiplier"]:
        if tier9_col in creatives_df.columns and tier9_col not in creative_cols:
            creative_cols.append(tier9_col)
    exposures = exposures.merge(
        creatives_df[[c for c in creative_cols if c in creatives_df.columns]],
        on="creative_id",
        how="left",
    )
    # Merge customer traits (and trust/desire for engagement; loyalty_propensity for 8-axis mastery)
    trait_cols = [
        "customer_id",
        "price_sensitivity",
        "impulse_level",
        "quality_expectation",
        "regret_propensity",
        "loyalty_propensity",
    ]
    if "trust_score" in active_customers.columns:
        trait_cols.append("trust_score")
    if "expressed_desire_level" in active_customers.columns:
        trait_cols.append("expressed_desire_level")
    exposures = exposures.merge(
        active_customers[[c for c in trait_cols if c in active_customers.columns]].drop_duplicates("customer_id"),
        on="customer_id",
        how="left",
    )

    # Vectorized click probability: trait–promise matches nudge probability, not dominate.
    # Legacy 4 axes
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
        (1.0 - exposures["regret_propensity"].fillna(0.5).values)
        * exposures["promise_safety_focus"].values
    )
    n_axes = 4
    scale = 0.02
    noise = rng.normal(0, 0.0025, size=len(exposures))
    match_avg = (match_discount + match_urgency + match_premium + match_safety) / n_axes
    click_prob = base_ctr + scale * match_avg + noise

    # --- Tier 9: angle emotional weight scales 8-axis contribution ---
    angle_emotional_weight = np.ones(len(exposures))
    angle_ctr_volatility = np.ones(len(exposures))
    if "creative_angle" in exposures.columns:
        for ang, params in _ANGLE_PARAMS.items():
            mask = (exposures["creative_angle"] == ang).values
            angle_emotional_weight[mask] = params["emotional_weight"]
            angle_ctr_volatility[mask] = params["ctr_volatility"]

    # 8-axis alignment: soft probabilistic nudges, scaled by angle emotional weight
    scale_8 = float(config.get("promise_8_click_scale", 0.012))
    impulse = exposures["impulse_level"].fillna(0.5).values
    quality = exposures["quality_expectation"].fillna(0.5).values
    regret = exposures["regret_propensity"].fillna(0.5).values
    loyalty = exposures["loyalty_propensity"].fillna(0.5).values if "loyalty_propensity" in exposures.columns else np.full(len(exposures), 0.5)
    extra_8 = np.zeros(len(exposures))
    if "promise_status_intensity" in exposures.columns:
        extra_8 += scale_8 * impulse * exposures["promise_status_intensity"].fillna(0).values
    if "promise_belonging_intensity" in exposures.columns:
        extra_8 += scale_8 * (1.0 - regret) * exposures["promise_belonging_intensity"].fillna(0).values
    if "promise_control_intensity" in exposures.columns:
        extra_8 += scale_8 * (1.0 - regret) * exposures["promise_control_intensity"].fillna(0).values
    if "promise_transformation_intensity" in exposures.columns:
        extra_8 += scale_8 * quality * exposures["promise_transformation_intensity"].fillna(0).values
    if "promise_relief_intensity" in exposures.columns:
        extra_8 += scale_8 * impulse * exposures["promise_relief_intensity"].fillna(0).values
    if "promise_novelty_intensity" in exposures.columns:
        extra_8 += scale_8 * impulse * exposures["promise_novelty_intensity"].fillna(0).values
    if "promise_mastery_intensity" in exposures.columns:
        extra_8 += 0.006 * loyalty * exposures["promise_mastery_intensity"].fillna(0).values
    extra_8 *= angle_emotional_weight
    click_prob = click_prob + extra_8

    # --- Tier 9: hook attention spike (short-lived CTR boost) ---
    if "creative_hook_pattern" in exposures.columns:
        hook_spike = np.zeros(len(exposures))
        for hook, params in _HOOK_PARAMS.items():
            mask = (exposures["creative_hook_pattern"] == hook).values
            hook_spike[mask] = params["attention_spike"]
        click_prob += hook_spike

    # --- Tier 9: angle CTR volatility (noise scale modifier) ---
    volatility_noise = rng.normal(0, 0.003, size=len(exposures)) * (angle_ctr_volatility - 1.0)
    click_prob += volatility_noise

    # --- Tier 9: persona resonance modifier (trait affinity → click lift/penalty) ---
    if "creative_persona" in exposures.columns:
        persona_resonance = np.full(len(exposures), 0.5)
        price_sens = exposures["price_sensitivity"].fillna(0.5).values
        for i in range(len(exposures)):
            persona = exposures["creative_persona"].iat[i]
            if pd.notna(persona):
                trait_vals = {
                    "price_sensitivity": float(price_sens[i]),
                    "impulse_level": float(impulse[i]),
                    "quality_expectation": float(quality[i]),
                    "regret_propensity": float(regret[i]),
                    "loyalty_propensity": float(loyalty[i]),
                }
                persona_resonance[i] = _persona_resonance(persona, trait_vals)
        persona_modifier = 0.015 * (persona_resonance - 0.5)
        click_prob += persona_modifier

    click_prob = np.clip(click_prob, 0.0, 1.0)

    # Tier 4 creative fatigue: effectiveness decays with repeated exposure (same customer–creative).
    fatigue_beta = float(config.get("fatigue_beta", 0.15))
    if "creative_fatigue_map" in customers_df.columns and fatigue_beta > 0:
        fatigue_rows = []
        for _, c in customers_df.iterrows():
            m = c.get("creative_fatigue_map")
            if not isinstance(m, dict):
                continue
            for crid, f in m.items():
                if float(f) >= 0.01:
                    fatigue_rows.append({"customer_id": c["customer_id"], "creative_id": crid, "fatigue": float(f)})
        if fatigue_rows:
            fatigue_df = pd.DataFrame(fatigue_rows)
            exposures = exposures.merge(fatigue_df, on=["customer_id", "creative_id"], how="left")
            exposures["fatigue"] = exposures["fatigue"].fillna(0)
        else:
            exposures["fatigue"] = np.zeros(len(exposures))

        # Tier 9: hook pattern accelerates fatigue sensitivity
        effective_fatigue = exposures["fatigue"].values.copy()
        if "creative_hook_pattern" in exposures.columns:
            hook_novelty_decay = np.ones(len(exposures))
            for hook, params in _HOOK_PARAMS.items():
                mask = (exposures["creative_hook_pattern"] == hook).values
                hook_novelty_decay[mask] = params["novelty_decay"]
            effective_fatigue *= hook_novelty_decay

        # Tier 9: persona-aligned humans fatigue slower
        if "creative_persona" in exposures.columns and "persona_resonance" in dir():
            fatigue_resistance = 1.0 - 0.25 * (persona_resonance - 0.5)
            effective_fatigue *= np.clip(fatigue_resistance, 0.5, 1.5)

        click_prob = click_prob * np.exp(-effective_fatigue * fatigue_beta)

        # Novelty promise: fatigue hits harder (repeated novelty loses appeal)
        if "promise_novelty_intensity" in exposures.columns:
            novelty = exposures["promise_novelty_intensity"].fillna(0).values
            fatigue_val = effective_fatigue
            click_prob = click_prob * (1.0 - 0.08 * novelty * np.minimum(fatigue_val, 2.0))
    # Upgrade 7: innate creative quality multiplier on click probability
    if "innate_quality" in exposures.columns:
        click_prob = click_prob * exposures["innate_quality"].fillna(1.0).values
    # Upgrade 9: creative lifecycle performance multiplier
    if "performance_multiplier" in exposures.columns:
        click_prob = click_prob * exposures["performance_multiplier"].fillna(1.0).values
    click_prob = np.clip(click_prob, 0.0, 1.0)

    clicked = (rng.random(size=len(exposures)) < click_prob).astype(int)
    exposures["clicked"] = clicked

    # Engagement signals (behavior before click): watch_time, liked, saved, shared. No interpretations.
    enable_engagement = config.get("engagement_signals", True)
    n_exp = len(exposures)
    alignment = (match_discount + match_urgency + match_premium + match_safety) / 4.0
    trust = exposures["trust_score"].fillna(0.5).values if "trust_score" in exposures.columns else np.full(n_exp, 0.5)
    desire = exposures["expressed_desire_level"].fillna(0.1).values if "expressed_desire_level" in exposures.columns else np.full(n_exp, 0.1)
    fatigue = exposures["fatigue"].fillna(0).values if "fatigue" in exposures.columns else np.zeros(n_exp)

    if enable_engagement:
        # repeat_count and promise_pressure for video retention model
        if "repeat_creative_exposure_count" not in exposures.columns:
            if exposure_count_map is not None:
                exposures["repeat_creative_exposure_count"] = exposures.apply(
                    lambda r: exposure_count_map.get((r["customer_id"], r["creative_id"]), 0), axis=1
                )
            elif "fatigue" in exposures.columns:
                exposures["repeat_creative_exposure_count"] = np.maximum(0, np.round(exposures["fatigue"].fillna(0).values * 2.0).astype(int))
            else:
                exposures["repeat_creative_exposure_count"] = 0
        repeat_count = exposures["repeat_creative_exposure_count"].fillna(0).values.astype(np.int64)
        exposures["_promise_pressure"] = exposures["creative_id"].map(lambda cid: _promise_pressure_from_creative(cid, creatives_df))
        pressure = exposures["_promise_pressure"].fillna(0).values.astype(np.float64)
        is_video = exposures["creative_type"].map(lambda ct: _is_video_type(ct) if pd.notna(ct) else False).values if "creative_type" in exposures.columns else np.zeros(n_exp, dtype=bool)
        duration = exposures["video_duration_seconds"].fillna(0).values.astype(np.float64) if "video_duration_seconds" in exposures.columns else np.zeros(n_exp)
        # Video: Beta(alpha, beta) watch_ratio; non-video: watch_time = 0
        base_alpha, base_beta = 1.2, 4.0
        alignment_watch_strength = float(config.get("alignment_watch_strength", 2.0))
        novelty_bonus = float(config.get("novelty_bonus", 0.8))
        fatigue_watch_penalty = float(config.get("fatigue_watch_penalty", 2.5))
        pressure_penalty = float(config.get("pressure_penalty", 1.5))
        trust_watch_bonus = float(config.get("trust_watch_bonus", 0.5))
        distrust_watch_penalty = float(config.get("distrust_watch_penalty", 0.5))
        alpha = base_alpha + alignment * alignment_watch_strength + np.where(repeat_count == 0, novelty_bonus, 0.0)
        beta = base_beta + fatigue * fatigue_watch_penalty + pressure * (1.0 - alignment) * pressure_penalty
        alpha = alpha + np.where(trust > 0.7, trust_watch_bonus, 0.0)
        beta = beta + np.where(trust < 0.3, distrust_watch_penalty, 0.0)

        # Tier 9: creative_type watch_time_mult — UGC/creator content holds attention differently
        ct_watch_mult = np.ones(n_exp)
        if "creative_type" in exposures.columns:
            for ct, params in _CREATIVE_TYPE_PARAMS.items():
                mask = (exposures["creative_type"] == ct).values
                if params["watch_time_mult"] > 0:
                    ct_watch_mult[mask] = params["watch_time_mult"]
        alpha *= np.where(is_video, ct_watch_mult, 1.0)

        alpha, beta = np.maximum(alpha, 0.1), np.maximum(beta, 0.1)
        watch_ratio = rng.beta(alpha, beta)
        watch_seconds = np.where(is_video & (duration > 0), np.minimum(watch_ratio * duration, duration), 0.0)
        exposures["watch_time_seconds"] = np.round(np.clip(watch_seconds, 0.0, 900.0)).astype(np.float64)
        exposures.drop(columns=["_promise_pressure"], inplace=True, errors="ignore")
        # Non-video: keep current watch (short “view” time); video already has realistic cap
        # liked: probability from alignment + trust
        p_like = np.clip(0.02 + 0.05 * alignment + 0.03 * trust, 0.0, 1.0)
        exposures["liked"] = (rng.random(n_exp) < p_like).astype(int)
        # saved: rarer; stronger alignment needed
        p_save = np.clip(0.005 + 0.03 * (alignment ** 2), 0.0, 1.0)
        exposures["saved"] = (rng.random(n_exp) < p_save).astype(int)
        # shared: rarest; high trust + high alignment
        p_share = np.clip(0.001 + 0.02 * alignment * trust, 0.0, 1.0)
        exposures["shared"] = (rng.random(n_exp) < p_share).astype(int)
    else:
        exposures["watch_time_seconds"] = 0.0
        exposures["liked"] = 0
        exposures["saved"] = 0
        exposures["shared"] = 0

    # Raw observability signals (breadcrumbs for dbt; no interpretations).
    exposures["multi_ad_session_depth"] = exposures.groupby("customer_id").cumcount() + 1
    # Deterministic within-day timestamp for tie-breaking (production logs would have an event timestamp)
    # 1st exposure of the day -> +0 minutes, 2nd -> +1 minute, etc.
    exposures["exposure_timestamp"] = exposures["date"] + pd.to_timedelta(
        exposures["multi_ad_session_depth"].astype(int) - 1, unit="m"
    )
    # engagement_intensity_score: 0–1 from watch/like/save/share (normalized)
    w = exposures["watch_time_seconds"].fillna(0).values / 30.0
    li = exposures["liked"].fillna(0).values
    sv = exposures["saved"].fillna(0).values
    sh = exposures["shared"].fillna(0).values
    exposures["engagement_intensity_score"] = np.clip(
        0.4 * w + 0.3 * li + 0.2 * sv + 0.1 * sh, 0.0, 1.0
    ).astype(np.float64)
    # repeat_creative_exposure_count: from map or fatigue proxy
    if exposure_count_map is not None:
        exposures["repeat_creative_exposure_count"] = exposures.apply(
            lambda r: exposure_count_map.get((r["customer_id"], r["creative_id"]), 0), axis=1
        )
    elif "fatigue" in exposures.columns:
        exposures["repeat_creative_exposure_count"] = np.maximum(
            0, np.round(exposures["fatigue"].fillna(0).values * 2.0).astype(int)
        )
    else:
        exposures["repeat_creative_exposure_count"] = 0
    # days_since_last_exposure: from last_exposure_by_customer (date or timestamp)
    if last_exposure_by_customer is not None:
        today = current_date.date() if hasattr(current_date, "date") else current_date
        def _days_since(cid: str) -> Optional[int]:
            last = last_exposure_by_customer.get(cid)
            if last is None or pd.isna(last):
                return None
            try:
                last_d = pd.to_datetime(last).normalize()
                last_d = last_d.date() if hasattr(last_d, "date") else last_d
                return (today - last_d).days
            except Exception:
                return None
        exposures["days_since_last_exposure"] = exposures["customer_id"].map(
            lambda cid: _days_since(cid) if isinstance(cid, str) else None
        )
    else:
        exposures["days_since_last_exposure"] = pd.NA

    out_cols = ["date", "customer_id", "ad_id", "creative_id", "impression", "clicked",
                "watch_time_seconds", "liked", "saved", "shared",
                "exposure_timestamp", "creative_type", "video_duration_seconds"] + _raw_signal_cols
    out = exposures[[c for c in out_cols if c in exposures.columns]].copy()
    return out


def update_creative_lifecycle(
    creatives_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Upgrade 9: Update creative lifecycle state based on cumulative impressions.

    Lifecycle stages:
      launch   (0 → novelty_threshold):   multiplier ramps 0.8 → 1.2
      ramp     (novelty → ramp_end):       multiplier = 1.2 (peak)
      plateau  (ramp_end → plateau_end):   multiplier = 1.0 (steady)
      fatigue  (plateau_end → exhausted):  multiplier decays exponentially
      exhausted (below floor):             multiplier = min_floor

    innate_quality modulates: higher quality = longer effective plateau.
    """
    cl = config.get("creative_lifecycle", {})
    if not cl.get("enabled", False):
        return creatives_df

    novelty_end = int(cl.get("novelty_bonus_duration_impressions", 5000))
    ramp_end = int(cl.get("ramp_end_impressions", 10000))
    plateau_end = int(cl.get("plateau_duration_impressions", 50000))
    fatigue_half_life = int(cl.get("fatigue_half_life_impressions", 100000))
    min_floor = float(cl.get("min_performance_floor", 0.2))

    if exposure_df.empty or "creative_id" not in exposure_df.columns:
        if "days_active" in creatives_df.columns:
            creatives_df = creatives_df.copy()
            creatives_df["days_active"] = creatives_df["days_active"].fillna(0).astype(int) + 1
        return creatives_df

    daily_counts = exposure_df.groupby("creative_id").size()

    creatives_df = creatives_df.copy()
    total_imps = creatives_df["total_impressions"].fillna(0).values.copy().astype(float)
    days_active = creatives_df["days_active"].fillna(0).values.copy().astype(int)
    quality = creatives_df["innate_quality"].fillna(1.0).values if "innate_quality" in creatives_df.columns else np.ones(len(creatives_df))

    cid_to_idx = {str(cid): i for i, cid in enumerate(creatives_df["creative_id"].values)}
    for cid, count in daily_counts.items():
        if str(cid) in cid_to_idx:
            total_imps[cid_to_idx[str(cid)]] += count

    days_active += 1

    stages = []
    multipliers = np.ones(len(creatives_df))

    for i in range(len(creatives_df)):
        imps = total_imps[i]
        q = quality[i]
        effective_plateau_end = plateau_end * q
        effective_fatigue_hl = fatigue_half_life * q

        if imps < novelty_end:
            stage = "launch"
            multipliers[i] = 0.8 + 0.4 * (imps / max(novelty_end, 1))
        elif imps < ramp_end:
            stage = "ramp"
            multipliers[i] = 1.2
        elif imps < effective_plateau_end:
            stage = "plateau"
            multipliers[i] = 1.0
        else:
            excess = imps - effective_plateau_end
            decay = np.exp(-np.log(2) * excess / max(effective_fatigue_hl, 1))
            mult = 1.0 * decay
            if mult < min_floor:
                stage = "exhausted"
                multipliers[i] = min_floor
            else:
                stage = "fatigue"
                multipliers[i] = mult
        stages.append(stage)

    creatives_df["total_impressions"] = total_imps.astype(int)
    creatives_df["days_active"] = days_active
    creatives_df["lifecycle_stage"] = stages
    creatives_df["performance_multiplier"] = np.round(multipliers, 4)

    return creatives_df
