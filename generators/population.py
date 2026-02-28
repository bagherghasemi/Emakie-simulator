"""
Time-Dependent Population Dynamics.

Manages the prospect pool over time: daily inflow of new prospects with
trait drift (later cohorts are harder to convert), culling of expired
unconverted prospects, and tracking of market saturation.

When population_dynamics is disabled or absent from config, the manager
is inert — it returns the original prospects_df unchanged.
"""

import uuid

import numpy as np
import pandas as pd

from generators.humans import FIRST_NAMES, LAST_NAMES, FAKE_REGIONS, FAKE_CITIES


class PopulationManager:
    """Mutable prospect pool with daily inflow/outflow."""

    def __init__(self, initial_prospects_df: pd.DataFrame, config: dict, sim_start_date):
        self.prospects_df = initial_prospects_df.copy()
        self._sim_start_date = pd.Timestamp(sim_start_date).normalize()
        self._base_seed = int(config.get("seed", 0))
        self._config = config

        pop = config.get("population_dynamics", {})
        self._enabled = pop.get("enabled", False)
        self._daily_inflow_base = int(pop.get("daily_inflow_base", 0))
        self._inflow_decay_rate = float(pop.get("inflow_decay_rate", 0.0005))
        self._quality_drift_rate = float(pop.get("quality_drift_rate", 0.0001))
        self._quality_drift_cap = float(pop.get("quality_drift_cap", 0.15))
        self._pool_expiry_days = int(pop.get("pool_expiry_days", 365))

        # Phase-dependent inflow multipliers
        self._phase_inflow_mult = pop.get("phase_inflow_mult", {
            "launch": 0.5, "growth": 1.2, "maturation": 0.8,
            "saturation": 0.5, "decline": 0.4,
        })
        # Monthly seasonality for inflow
        self._month_inflow_mult = pop.get("month_inflow_mult", {})

        # Market size for saturation tracking
        bl = config.get("brand_lifecycle", {})
        self._market_size = int(bl.get("market_size", pop.get("market_size", 500000)))

        # Counters
        self._total_generated = len(initial_prospects_df)
        self._total_converted = 0
        self._total_culled = 0

        # Track when each prospect entered the pool (for expiry)
        if "_pool_entry_date" not in self.prospects_df.columns:
            self.prospects_df["_pool_entry_date"] = self._sim_start_date

    def daily_update(self, current_date, brand_state=None) -> pd.DataFrame:
        """Add new prospects with drifted traits, cull expired ones.
        Returns the updated prospects_df."""
        if not self._enabled:
            return self.prospects_df

        current_date = pd.Timestamp(current_date).normalize()
        days_elapsed = (current_date - self._sim_start_date).days

        # 1. Cull expired prospects (unconverted for > pool_expiry_days)
        if "_pool_entry_date" in self.prospects_df.columns and len(self.prospects_df) > 0:
            age = (current_date - self.prospects_df["_pool_entry_date"]).dt.days
            expired_mask = age > self._pool_expiry_days
            n_culled = int(expired_mask.sum())
            if n_culled > 0:
                self.prospects_df = self.prospects_df[~expired_mask].copy()
                self._total_culled += n_culled

        # 2. Compute daily inflow count
        n_new = self._compute_inflow(days_elapsed, current_date, brand_state)

        # 3. Generate new prospects with drifted traits
        if n_new > 0:
            new_prospects = self._generate_drifted_prospects(
                n_new, days_elapsed, current_date
            )
            self.prospects_df = pd.concat(
                [self.prospects_df, new_prospects], ignore_index=True
            )
            self._total_generated += n_new

        return self.prospects_df

    def remove_converted(self, converted_ids: set):
        """Remove prospects that became customers."""
        if not converted_ids:
            return
        mask = self.prospects_df["customer_id"].astype(str).isin(
            {str(cid) for cid in converted_ids}
        )
        n_converted = int(mask.sum())
        if n_converted > 0:
            self.prospects_df = self.prospects_df[~mask].copy()
            self._total_converted += n_converted

    @property
    def saturation(self) -> float:
        """total_converted / market_size."""
        return self._total_converted / max(self._market_size, 1)

    def _compute_inflow(self, days_elapsed: int, current_date, brand_state) -> int:
        """Compute how many new prospects arrive today."""
        if self._daily_inflow_base <= 0:
            return 0

        # Base inflow with exponential decay (easy audience exhausted)
        inflow = self._daily_inflow_base * np.exp(
            -self._inflow_decay_rate * days_elapsed
        )

        # Phase multiplier from BrandState
        if brand_state is not None:
            phase = brand_state.lifecycle_phase.value
            inflow *= float(self._phase_inflow_mult.get(phase, 1.0))

        # Monthly seasonality
        month = current_date.month
        inflow *= float(self._month_inflow_mult.get(month, self._month_inflow_mult.get(str(month), 1.0)))

        return max(0, int(round(inflow)))

    def _generate_drifted_prospects(
        self, n: int, days_elapsed: int, current_date
    ) -> pd.DataFrame:
        """Generate n new prospects with trait distributions drifted by time."""
        day_ordinal = current_date.toordinal()
        rng = np.random.default_rng((self._base_seed + day_ordinal) % (2**31))

        # Trait drift: later cohorts have lower loyalty, higher price sensitivity
        drift = min(days_elapsed * self._quality_drift_rate, self._quality_drift_cap)

        mean_base = 0.5
        std = 0.2

        latent = rng.normal(mean_base, std, size=(n, 5))
        latent = np.clip(latent, 0.0, 1.0)

        price_sensitivity = latent[:, 0] + drift  # higher = harder to convert
        impulse_level = latent[:, 1] - drift * 0.5
        loyalty_propensity = latent[:, 2] - drift  # lower = less sticky
        regret_propensity = latent[:, 3]
        quality_expectation = latent[:, 4]

        # Apply trait correlations (same as humans.py)
        regret_propensity = np.clip(
            regret_propensity
            + 0.12 * impulse_level
            - 0.12 * loyalty_propensity
            + 0.10 * quality_expectation
            + rng.normal(0, 0.05, size=n),
            0.0, 1.0,
        )
        price_sensitivity = np.clip(price_sensitivity, 0.0, 1.0)
        impulse_level = np.clip(impulse_level, 0.0, 1.0)
        loyalty_propensity = np.clip(loyalty_propensity, 0.0, 1.0)

        # Identity
        prospect_ids = [
            f"prospect_{self._base_seed}_dyn_{self._total_generated + i}"
            for i in range(n)
        ]

        income_levels = np.array(["low", "mid", "high"])
        income_probs = np.array([0.25, 0.55, 0.20])
        income_proxy = rng.choice(income_levels, size=n, p=income_probs)

        channels = np.array(["meta", "organic", "search"])
        channel_probs = np.array([0.35, 0.35, 0.30])
        acq_channel = rng.choice(channels, size=n, p=channel_probs)

        signup_dates = [pd.Timestamp("1900-01-01")] * n

        # Upgrade 7: spending propensity for new prospects
        spending_sigma = float(self._config.get("distribution_realism", {}).get("spending_propensity_sigma", 0.5))
        spending_propensity = np.clip(rng.lognormal(0, spending_sigma, size=n), 0.2, 5.0)

        df = pd.DataFrame({
            "id": prospect_ids,
            "created_at": signup_dates,
            "price_sensitivity": price_sensitivity,
            "impulse_level": impulse_level,
            "loyalty_propensity": loyalty_propensity,
            "regret_propensity": regret_propensity,
            "quality_expectation": quality_expectation,
            "income_proxy": income_proxy,
            "acquisition_channel_preference": acq_channel,
            "spending_propensity": spending_propensity,
            "trust_score": np.full(n, 0.5),
            "disappointment_memory": np.zeros(n),
            "satisfaction_memory": np.zeros(n),
            "recent_negative_velocity": np.zeros(n),
            "discount_dependency": np.zeros(n),
            "exposure_count": np.zeros(n, dtype=np.int64),
            "expressed_desire_level": np.full(n, 0.1),
            "desire_decay_memory": np.zeros(n),
            "creative_fatigue_map": [{} for _ in range(n)],
            "cart_memory": [None] * n,
            "shipping_bad_count": np.zeros(n, dtype=np.int64),
            "shipping_good_count": np.zeros(n, dtype=np.int64),
            "discount_only_buyer": np.zeros(n, dtype=bool),
            "days_since_last_interaction": np.zeros(n, dtype=np.int64),
            "last_attributed_ad_id": [None] * n,
            "last_attributed_adset_id": [None] * n,
            "last_attributed_campaign_id": [None] * n,
            "last_attributed_creative_id": [None] * n,
            "last_exposure_date": [pd.NaT] * n,
        })
        df["last_order_date"] = pd.NaT
        df["updated_at"] = df["created_at"]
        df["state"] = "enabled"
        df["verified_email"] = True
        df["email"] = "prospect_dyn_" + pd.Series(range(n)).astype(str) + "@example.com"
        df["first_name"] = rng.choice(FIRST_NAMES, size=n).tolist()
        df["last_name"] = rng.choice(LAST_NAMES, size=n).tolist()
        df["accepts_marketing"] = rng.random(size=n) < 0.70
        df["default_address_country"] = rng.choice(
            ["US", "CA", "UK", "DE"], size=n
        ).tolist()
        df["default_address_region"] = rng.choice(FAKE_REGIONS, size=n).tolist()
        df["default_address_city"] = rng.choice(FAKE_CITIES, size=n).tolist()
        df["last_order_at"] = pd.NaT
        df["tax_exempt"] = pd.array([pd.NA] * n, dtype="boolean")
        for col in [
            "phone", "locale", "tags", "note", "source_name",
            "display_name", "default_address_id", "default_address_postal_code",
        ]:
            df[col] = None
        df["customer_id"] = df["id"].values
        df["signup_date"] = df["created_at"].values
        df["prospect_id"] = df["id"].values
        df["_pool_entry_date"] = current_date

        return df
