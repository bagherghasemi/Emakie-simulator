"""
Momentum, Inertia & Autocorrelation.

Adds temporal autocorrelation to key daily metrics (revenue, CTR, CVR) via
EMA-based momentum tracking, plus weekly/monthly seasonality multipliers.

When momentum is disabled or absent from config, all multipliers return 1.0.
"""

import numpy as np


class MomentumTracker:
    """EMA-based autocorrelation for key daily metrics."""

    def __init__(self, config: dict):
        mom = config.get("momentum", {})
        self._enabled = mom.get("enabled", False)
        self._weights = {
            "revenue": float(mom.get("revenue_autocorrelation", 0.7)),
            "ctr": float(mom.get("ctr_autocorrelation", 0.5)),
            "cvr": float(mom.get("conversion_autocorrelation", 0.6)),
        }
        self._ema = {}  # metric_name -> current EMA value
        self._baseline = {}  # metric_name -> first EMA value (for rate_modifier)

        # Seasonality patterns
        seasonality = mom.get("seasonality", {})
        self._weekly_pattern = seasonality.get(
            "weekly_pattern", [1.0] * 7
        )  # Mon-Sun
        self._monthly_pattern = seasonality.get(
            "monthly_pattern", [1.0] * 12
        )  # Jan-Dec

    def update(self, actuals: dict):
        """Feed today's actual metrics to update EMAs."""
        if not self._enabled:
            return
        for key, val in actuals.items():
            if key in self._weights and val is not None:
                w = self._weights[key]
                prev = self._ema.get(key)
                if prev is None:
                    self._ema[key] = float(val)
                else:
                    self._ema[key] = w * prev + (1 - w) * float(val)

    def get_multiplier(self, metric: str, today_raw: float) -> float:
        """Return momentum-adjusted multiplier: pulls toward recent EMA."""
        if not self._enabled or metric not in self._ema:
            return 1.0
        if today_raw <= 0:
            return 1.0
        ema_val = self._ema[metric]
        if ema_val <= 0:
            return 1.0
        # Blend: momentum multiplier pulls today's value toward EMA
        return ema_val / today_raw

    def get_rate_modifier(self, metric: str) -> float:
        """
        Return a rate modifier based on recent momentum.

        Uses the EMA relative to its own running mean as a multiplier.
        If recent EMA is above long-term average → modifier > 1.0 (boost).
        If below → modifier < 1.0 (dampen). Clamped to [0.7, 1.3].
        """
        if not self._enabled or metric not in self._ema:
            return 1.0
        ema_val = self._ema[metric]
        if ema_val <= 0:
            return 1.0
        # Use EMA as direct multiplier vs a neutral baseline of 1.0
        # The EMA itself tracks the metric; for rates like CTR/CVR,
        # the modifier = 1.0 + (ema - baseline) * sensitivity
        # where baseline is the initial EMA value stored at first update
        if metric not in self._baseline:
            self._baseline[metric] = ema_val
            return 1.0
        baseline = self._baseline[metric]
        if baseline <= 0:
            return 1.0
        ratio = ema_val / baseline
        return max(0.7, min(1.3, ratio))

    def get_seasonality_multiplier(self, date) -> float:
        """Return combined weekly × monthly seasonality multiplier."""
        if not self._enabled:
            return 1.0
        try:
            weekday = date.weekday()  # 0=Mon, 6=Sun
            month = date.month - 1    # 0-indexed
            weekly = float(self._weekly_pattern[weekday]) if weekday < len(self._weekly_pattern) else 1.0
            monthly = float(self._monthly_pattern[month]) if month < len(self._monthly_pattern) else 1.0
            return weekly * monthly
        except (AttributeError, IndexError):
            return 1.0
