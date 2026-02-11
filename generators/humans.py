"""
Customer population generator for behavioral e-commerce simulation.

Produces a DataFrame of synthetic customers with stable latent traits and
behavioral anchors. These fields are hidden variables used later by simulation
mechanics (e.g. propensity to buy on impulse, sensitivity to price) and should
not be exposed as "observed" data—they drive event generation.
"""

import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


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
    - Scenario anchors (booleans): is_whale (top ~5% loyalty_propensity and
      bottom ~30% price_sensitivity—high repeat buyers), is_fragile (high
      regret_propensity and high quality_expectation—return/attrition risk).

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
        One row per customer with columns: customer_id, signup_date,
        price_sensitivity, impulse_level, loyalty_propensity, regret_propensity,
        quality_expectation, income_proxy, acquisition_channel_preference,
        is_whale, is_fragile.
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

    # Scenario anchors: whale (high repeat potential), fragile (return/attrition risk)
    loyalty_p95 = np.percentile(loyalty_propensity, 95)
    price_p30 = np.percentile(price_sensitivity, 30)
    is_whale = (loyalty_propensity >= loyalty_p95) & (price_sensitivity <= price_p30)

    regret_p70 = np.percentile(regret_propensity, 70)
    quality_p70 = np.percentile(quality_expectation, 70)
    is_fragile = (regret_propensity >= regret_p70) & (quality_expectation >= quality_p70)

    # Behavioral anchors: income_proxy with realistic skew (more mid)
    income_levels = np.array(["low", "mid", "high"])
    income_probs = np.array([0.25, 0.55, 0.20])  # more mid
    income_proxy = rng.choice(income_levels, size=n, p=income_probs)

    # Acquisition channel preference
    channels = np.array(["meta", "organic", "search"])
    channel_probs = np.array([0.35, 0.35, 0.30])  # can be tuned
    acquisition_channel_preference = rng.choice(channels, size=n, p=channel_probs)

    return pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": signup_dates,
        "price_sensitivity": price_sensitivity,
        "impulse_level": impulse_level,
        "loyalty_propensity": loyalty_propensity,
        "regret_propensity": regret_propensity,
        "quality_expectation": quality_expectation,
        "income_proxy": income_proxy,
        "acquisition_channel_preference": acquisition_channel_preference,
        "is_whale": is_whale,
        "is_fragile": is_fragile,
    })