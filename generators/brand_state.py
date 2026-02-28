"""
Brand Lifecycle State Manager.

Tracks the brand's lifecycle phase and accumulates brand-level metrics that
affect all downstream systems. The BrandState evolves daily based on emergent
metrics (revenue, orders, refunds, customers) — phase transitions are NOT
time-scripted but emerge from accumulated performance data.

Phases: LAUNCH → GROWTH → MATURATION → SATURATION → DECLINE
Each phase applies multiplicative effects to CTR, CVR, refund rate, repeat
rate, and CPA. All multipliers default to 1.0 when brand_lifecycle is
disabled or absent from config, preserving backward compatibility.
"""

from collections import deque
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class BrandPhase(Enum):
    LAUNCH = "launch"
    GROWTH = "growth"
    MATURATION = "maturation"
    SATURATION = "saturation"
    DECLINE = "decline"


# Default phase effects (all overridable via config)
_DEFAULT_PHASE_EFFECTS = {
    BrandPhase.LAUNCH: {
        "ctr_mult": 0.85, "cvr_mult": 0.75, "refund_mult": 1.1,
        "repeat_mult": 0.5, "cpa_mult": 0.9,
    },
    BrandPhase.GROWTH: {
        "ctr_mult": 1.1, "cvr_mult": 1.05, "refund_mult": 0.9,
        "repeat_mult": 1.2, "cpa_mult": 1.0,
    },
    BrandPhase.MATURATION: {
        "ctr_mult": 1.0, "cvr_mult": 1.1, "refund_mult": 0.85,
        "repeat_mult": 1.4, "cpa_mult": 1.15,
    },
    BrandPhase.SATURATION: {
        "ctr_mult": 0.9, "cvr_mult": 0.85, "refund_mult": 1.0,
        "repeat_mult": 1.1, "cpa_mult": 1.4,
    },
    BrandPhase.DECLINE: {
        "ctr_mult": 0.8, "cvr_mult": 0.7, "refund_mult": 1.3,
        "repeat_mult": 0.7, "cpa_mult": 1.6,
    },
}

# Neutral effects (when brand_lifecycle is disabled)
_NEUTRAL_EFFECTS = {
    "ctr_mult": 1.0, "cvr_mult": 1.0, "refund_mult": 1.0,
    "repeat_mult": 1.0, "cpa_mult": 1.0,
}


class BrandState:
    """Mutable brand-level state, updated daily in main.py."""

    def __init__(self, config: dict):
        bl = config.get("brand_lifecycle", {})
        self._enabled = bl.get("enabled", False)

        # Current state
        self.brand_age_days: int = 0
        self.lifecycle_phase: BrandPhase = BrandPhase(
            bl.get("initial_phase", "launch")
        )

        # Accumulators
        self.cumulative_revenue: float = 0.0
        self.cumulative_orders: int = 0
        self.cumulative_customers: int = 0
        self.cumulative_refunds: int = 0
        self.cumulative_repeat_orders: int = 0
        self.brand_trust_baseline: float = 0.5
        self.market_penetration: float = 0.0
        self.acquisition_difficulty_multiplier: float = 1.0
        self.discount_dependency_score: float = 0.0
        self.creative_fatigue_baseline: float = 0.0

        # Feedback loop state (Upgrade 5)
        self.acquisition_pressure: float = 0.0
        self.discount_pressure: float = 0.0

        # Long-term accumulators (Upgrade 10)
        self.cumulative_creative_churn: int = 0
        self.cumulative_discount_usage_pct: float = 0.0
        self.cumulative_refund_rate_ema: float = 0.0
        self.cumulative_acquisition_spend: float = 0.0
        self.brand_reputation_score: float = 0.5
        self.audience_exhaustion_index: float = 0.0

        # Rolling windows for transition detection
        self._revenue_window: deque = deque(maxlen=30)
        self._order_velocity_window: deque = deque(maxlen=14)
        self._repeat_rate_window: deque = deque(maxlen=14)
        self._cpa_window: deque = deque(maxlen=30)
        self._refund_rate_window: deque = deque(maxlen=30)

        # Transition thresholds from config
        self._market_size = int(bl.get("market_size", 500000))
        self._growth_velocity_threshold = float(
            bl.get("growth_velocity_threshold", 10)
        )
        self._growth_repeat_stability_cv = float(
            bl.get("growth_repeat_stability_cv", 0.3)
        )
        self._maturation_penetration_pct = float(
            bl.get("maturation_penetration_pct", 0.20)
        )
        self._maturation_cpa_rise_pct = float(
            bl.get("maturation_cpa_rise_pct", 0.50)
        )
        self._saturation_cpa_ltv_ratio = float(
            bl.get("saturation_cpa_ltv_ratio", 0.8)
        )
        self._decline_repeat_decay_pct = float(
            bl.get("decline_repeat_decay_pct", 0.30)
        )
        self._decline_cpa_multiplier = float(
            bl.get("decline_cpa_multiplier", 2.0)
        )

        # Phase effects from config (or defaults)
        self._phase_effects: Dict[BrandPhase, Dict[str, float]] = {}
        config_effects = bl.get("phase_effects", {})
        for phase in BrandPhase:
            if phase.value in config_effects:
                self._phase_effects[phase] = {
                    **_DEFAULT_PHASE_EFFECTS[phase],
                    **config_effects[phase.value],
                }
            else:
                self._phase_effects[phase] = _DEFAULT_PHASE_EFFECTS[phase].copy()

        # Feedback loop config (Upgrade 5)
        fl = config.get("feedback_loops", {})
        self._feedback_enabled = fl.get("enabled", False)
        self._pressure_ramp_rate = float(fl.get("pressure_ramp_rate", 0.02))
        self._pressure_decay_rate = float(fl.get("pressure_decay_rate", 0.01))
        self._revenue_target_window = int(fl.get("revenue_target_window_days", 30))
        self._acq_difficulty_growth_rate = float(fl.get("acquisition_difficulty_growth_rate", 0.001))

        # Long-term accumulator config (Upgrade 10)
        lt = config.get("long_term_accumulators", {})
        self._ctr_erosion_per_churn = float(lt.get("ctr_erosion_per_churn", 0.02))
        self._discount_cvr_penalty = float(lt.get("discount_dependency_cvr_penalty", 0.3))
        self._trust_floor_from_refunds = float(lt.get("trust_floor_from_refunds", 0.5))
        self._exhaustion_cpa_mult = float(lt.get("exhaustion_cpa_multiplier", 1.5))

        # Track launch-phase CPA for comparison
        self._launch_cpa_samples: list = []
        self._launch_cpa_avg: Optional[float] = None
        self._peak_repeat_rate: float = 0.0

    def get_phase_effects(self) -> Dict[str, float]:
        """Return multiplier dict for current phase. All 1.0 if disabled."""
        if not self._enabled:
            return _NEUTRAL_EFFECTS.copy()

        effects = self._phase_effects.get(
            self.lifecycle_phase, _NEUTRAL_EFFECTS
        ).copy()

        # Apply long-term accumulator modifications (Upgrade 10)
        if self.cumulative_creative_churn > 0:
            erosion = max(0.5, 1.0 - self._ctr_erosion_per_churn * self.cumulative_creative_churn)
            effects["ctr_mult"] *= erosion

        if self.audience_exhaustion_index > 0.3:
            exhaust_penalty = 1.0 + (self.audience_exhaustion_index - 0.3) * (self._exhaustion_cpa_mult - 1.0)
            effects["cpa_mult"] *= exhaust_penalty

        # Discount dependency CVR penalty: heavy discount usage erodes full-price conversion
        if self.cumulative_discount_usage_pct > 0.1:
            cvr_penalty = max(0.5, 1.0 - self._discount_cvr_penalty * self.cumulative_discount_usage_pct)
            effects["cvr_mult"] *= cvr_penalty

        return effects

    def get_trust_floor(self) -> float:
        """Trust floor derived from cumulative refund rate. High refunds = lower trust floor."""
        floor = max(0.1, self._trust_floor_from_refunds - self.cumulative_refund_rate_ema)
        return floor

    def update_daily(
        self,
        day_metrics: Dict[str, Any],
        customers_df=None,
        market_size: Optional[int] = None,
    ):
        """
        Called once per day after all events. Updates accumulators and
        checks for emergent phase transitions.

        day_metrics keys: revenue, orders, refunds, new_customers,
                         repeat_orders, spend, discount_orders
        """
        if not self._enabled:
            return

        self.brand_age_days += 1

        revenue = float(day_metrics.get("revenue", 0))
        orders = int(day_metrics.get("orders", 0))
        refunds = int(day_metrics.get("refunds", 0))
        new_customers = int(day_metrics.get("new_customers", 0))
        repeat_orders = int(day_metrics.get("repeat_orders", 0))
        spend = float(day_metrics.get("spend", 0))
        discount_orders = int(day_metrics.get("discount_orders", 0))

        # Update cumulative accumulators
        self.cumulative_revenue += revenue
        self.cumulative_orders += orders
        self.cumulative_refunds += refunds
        self.cumulative_customers += new_customers
        self.cumulative_repeat_orders += repeat_orders
        self.cumulative_acquisition_spend += spend

        # Rolling windows
        self._revenue_window.append(revenue)
        self._order_velocity_window.append(orders)
        refund_rate = refunds / max(orders, 1)
        self._refund_rate_window.append(refund_rate)

        # Repeat rate
        repeat_rate = repeat_orders / max(orders, 1) if orders > 0 else 0
        self._repeat_rate_window.append(repeat_rate)

        # CPA
        cpa = spend / max(new_customers, 1) if new_customers > 0 else 0
        if cpa > 0:
            self._cpa_window.append(cpa)

        # Market penetration
        if market_size is None:
            market_size = self._market_size
        self.market_penetration = self.cumulative_customers / max(market_size, 1)
        self.audience_exhaustion_index = self.market_penetration

        # Acquisition difficulty: grows linearly as easy prospects are exhausted
        # multiplier = 1 + rate * days, making CVR decline over time
        self.acquisition_difficulty_multiplier = (
            1.0 + self._acq_difficulty_growth_rate * self.brand_age_days
        )

        # Track launch CPA for comparison in transitions
        # Collect CPA samples even after LAUNCH ends if baseline not yet established
        if (self.lifecycle_phase == BrandPhase.LAUNCH or self._launch_cpa_avg is None) and cpa > 0:
            self._launch_cpa_samples.append(cpa)
            if len(self._launch_cpa_samples) >= 7:
                self._launch_cpa_avg = np.mean(self._launch_cpa_samples)

        # Track peak repeat rate
        if len(self._repeat_rate_window) >= 7:
            avg_repeat = np.mean(list(self._repeat_rate_window))
            self._peak_repeat_rate = max(self._peak_repeat_rate, avg_repeat)

        # Update brand trust baseline from customer trust scores
        if customers_df is not None and "trust_score" in customers_df.columns:
            trust_vals = customers_df["trust_score"].dropna()
            if len(trust_vals) > 0:
                # EMA: slow-moving brand trust
                current_trust = float(trust_vals.mean())
                self.brand_trust_baseline = (
                    0.95 * self.brand_trust_baseline + 0.05 * current_trust
                )

        # Update cumulative refund rate EMA
        self.cumulative_refund_rate_ema = (
            0.95 * self.cumulative_refund_rate_ema + 0.05 * refund_rate
        )

        # Update discount usage percentage
        if orders > 0:
            day_disc_pct = discount_orders / orders
            self.cumulative_discount_usage_pct = (
                0.95 * self.cumulative_discount_usage_pct + 0.05 * day_disc_pct
            )

        # Update brand reputation
        if customers_df is not None and "satisfaction_memory" in customers_df.columns:
            sat = customers_df["satisfaction_memory"].fillna(0).mean()
            dis = customers_df.get("disappointment_memory", 0)
            if hasattr(dis, "mean"):
                dis = dis.fillna(0).mean()
            else:
                dis = 0
            net = (sat - dis) / max(sat + dis, 1)
            self.brand_reputation_score = (
                0.95 * self.brand_reputation_score + 0.05 * net
            )

        # Feedback loop updates (Upgrade 5)
        if self._feedback_enabled:
            self._update_feedback_loops(revenue)

        # Check phase transitions
        self._check_transitions()

    def _update_feedback_loops(self, today_revenue: float):
        """Update acquisition and discount pressure based on revenue trends."""
        if len(self._revenue_window) < self._revenue_target_window:
            return

        rolling_avg = np.mean(list(self._revenue_window))
        if rolling_avg > 0 and today_revenue < rolling_avg * 0.8:
            self.acquisition_pressure = min(
                self.acquisition_pressure + self._pressure_ramp_rate, 1.0
            )
        else:
            self.acquisition_pressure = max(
                self.acquisition_pressure - self._pressure_decay_rate, 0.0
            )

        # Discount pressure: rises when conversion is below historical average
        if len(self._order_velocity_window) >= 7:
            recent_velocity = np.mean(list(self._order_velocity_window)[-7:])
            full_velocity = np.mean(list(self._order_velocity_window))
            if full_velocity > 0 and recent_velocity < full_velocity * 0.7:
                self.discount_pressure = min(
                    self.discount_pressure + self._pressure_ramp_rate * 0.5, 1.0
                )
            else:
                self.discount_pressure = max(
                    self.discount_pressure - self._pressure_decay_rate * 0.5, 0.0
                )

    def _check_transitions(self):
        """Check for emergent phase transitions."""
        if self.lifecycle_phase == BrandPhase.LAUNCH:
            self._check_launch_to_growth()
        elif self.lifecycle_phase == BrandPhase.GROWTH:
            self._check_growth_to_maturation()
        elif self.lifecycle_phase == BrandPhase.MATURATION:
            self._check_maturation_to_saturation()
        elif self.lifecycle_phase == BrandPhase.SATURATION:
            self._check_saturation_to_decline()

    def _check_launch_to_growth(self):
        """LAUNCH → GROWTH: order velocity > threshold AND repeat stabilizes."""
        if len(self._order_velocity_window) < 14:
            return
        avg_velocity = np.mean(list(self._order_velocity_window))
        if avg_velocity < self._growth_velocity_threshold:
            return
        # Repeat rate stability
        if len(self._repeat_rate_window) < 7:
            return
        rr_values = list(self._repeat_rate_window)
        rr_mean = np.mean(rr_values)
        rr_cv = np.std(rr_values) / max(rr_mean, 0.001)
        if rr_cv <= self._growth_repeat_stability_cv:
            self.lifecycle_phase = BrandPhase.GROWTH

    def _check_growth_to_maturation(self):
        """GROWTH → MATURATION: penetration > threshold AND CPA risen."""
        if self.market_penetration < self._maturation_penetration_pct:
            return
        if self._launch_cpa_avg is None or len(self._cpa_window) < 14:
            return
        current_cpa = np.mean(list(self._cpa_window)[-14:])
        cpa_rise = (current_cpa - self._launch_cpa_avg) / max(
            self._launch_cpa_avg, 0.01
        )
        if cpa_rise >= self._maturation_cpa_rise_pct:
            self.lifecycle_phase = BrandPhase.MATURATION

    def _check_maturation_to_saturation(self):
        """MATURATION → SATURATION: CPA exceeds LTV ratio threshold."""
        if len(self._cpa_window) < 14:
            return
        current_cpa = np.mean(list(self._cpa_window)[-14:])
        avg_ltv = (
            self.cumulative_revenue / max(self.cumulative_customers, 1)
        )
        if avg_ltv > 0 and current_cpa / avg_ltv >= self._saturation_cpa_ltv_ratio:
            self.lifecycle_phase = BrandPhase.SATURATION

    def _check_saturation_to_decline(self):
        """SATURATION → DECLINE: repeat drops AND CPA is prohibitive."""
        if len(self._repeat_rate_window) < 7:
            return
        current_repeat = np.mean(list(self._repeat_rate_window))
        if self._peak_repeat_rate > 0:
            repeat_decay = (
                self._peak_repeat_rate - current_repeat
            ) / self._peak_repeat_rate
        else:
            repeat_decay = 0

        if repeat_decay < self._decline_repeat_decay_pct:
            return

        if self._launch_cpa_avg is not None and len(self._cpa_window) >= 7:
            current_cpa = np.mean(list(self._cpa_window)[-7:])
            if current_cpa >= self._launch_cpa_avg * self._decline_cpa_multiplier:
                self.lifecycle_phase = BrandPhase.DECLINE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/debugging."""
        return {
            "brand_age_days": self.brand_age_days,
            "lifecycle_phase": self.lifecycle_phase.value,
            "brand_trust_baseline": round(self.brand_trust_baseline, 4),
            "market_penetration": round(self.market_penetration, 6),
            "cumulative_revenue": round(self.cumulative_revenue, 2),
            "cumulative_orders": self.cumulative_orders,
            "cumulative_customers": self.cumulative_customers,
            "acquisition_pressure": round(self.acquisition_pressure, 4),
            "discount_pressure": round(self.discount_pressure, 4),
        }
