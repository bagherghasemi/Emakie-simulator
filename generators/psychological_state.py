"""
Lightweight psychological state per customer (Tier 2 behavioral causality).

Creatives influence future behavior through expectation, experience, and memory.
State: trust_score, disappointment_memory, satisfaction_memory.
Flow: expectation (from creative) vs experience (from order) → mismatch → update memory → trust.
Trust then influences refund probability and conversion (repeat) probability.
"""

import numpy as np
import pandas as pd

# Tier 9 imports (deferred to avoid circular import at module level)
_ANGLE_PARAMS = None
_PERSONA_TRAIT_AFFINITY = None
_CREATIVE_TYPE_PARAMS = None


def _get_tier9_params():
    """Lazy import of Tier 9 parameters from meta.py to avoid circular imports."""
    global _ANGLE_PARAMS, _PERSONA_TRAIT_AFFINITY, _CREATIVE_TYPE_PARAMS
    if _ANGLE_PARAMS is None:
        from .meta import (
            _ANGLE_PARAMS as AP, _PERSONA_TRAIT_AFFINITY as PTA,
            _CREATIVE_TYPE_PARAMS as CTP,
        )
        _ANGLE_PARAMS = AP
        _PERSONA_TRAIT_AFFINITY = PTA
        _CREATIVE_TYPE_PARAMS = CTP
    return _ANGLE_PARAMS, _PERSONA_TRAIT_AFFINITY, _CREATIVE_TYPE_PARAMS


# Promise axes on creatives (must match meta.py)
PROMISE_AXES = [
    "promise_discount_focus",
    "promise_premium_focus",
    "promise_urgency_focus",
    "promise_safety_focus",
]

# Canonical 8 psychological axes (when present on creatives)
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

# Expectation: 8-axis weights; transformation/status raise expectation most
EXPECTATION_8_WEIGHTS = {
    "promise_status_intensity": 1.4,
    "promise_safety_intensity": 0.7,
    "promise_control_intensity": 0.8,
    "promise_belonging_intensity": 0.9,
    "promise_transformation_intensity": 1.5,
    "promise_relief_intensity": 0.8,
    "promise_novelty_intensity": 0.9,
    "promise_mastery_intensity": 1.0,
}

# Pressure: all 8 contribute to disappointment amplification when expectation > experience
PRESSURE_8_WEIGHTS = {
    "promise_status_intensity": 1.2,
    "promise_safety_intensity": 0.8,
    "promise_control_intensity": 0.9,
    "promise_belonging_intensity": 1.0,
    "promise_transformation_intensity": 1.3,
    "promise_relief_intensity": 0.7,
    "promise_novelty_intensity": 0.9,
    "promise_mastery_intensity": 1.0,
}

# Quality level → numeric for experience score
_QUALITY_TO_SCORE = {"low": 0.25, "mid": 0.5, "high": 0.75}

# Tier 5: promise pressure (legacy 4 axes: discount, premium, urgency; safety not in pressure)
PROMISE_PRESSURE_WEIGHTS = (0.25, 0.40, 0.35)


def _promise_pressure_from_creative(creative_id, creatives_df: pd.DataFrame) -> float:
    """Tier 5: derived pressure from creative promise axes. High = high expectation, worse when oversold. Uses 8 axes when present."""
    if creatives_df.empty or creative_id is None or (isinstance(creative_id, float) and np.isnan(creative_id)):
        return 0.0
    row = creatives_df[creatives_df["creative_id"] == creative_id]
    if row.empty:
        return 0.0
    row = row.iloc[0]
    # 8-axis pressure when present
    have_8 = any(ax in row.index and pd.notna(row.get(ax)) for ax in PROMISE_8_AXES)
    if have_8:
        total, wsum = 0.0, 0.0
        for ax in PROMISE_8_AXES:
            if ax not in row.index:
                continue
            v = row[ax]
            if pd.notna(v) and isinstance(v, (int, float)):
                w = PRESSURE_8_WEIGHTS.get(ax, 1.0)
                total += float(v) * w
                wsum += w
        if wsum > 0:
            return float(np.clip(total / wsum, 0.0, 1.0))
    # Legacy 4-axis
    discount = row.get("promise_discount_focus")
    premium = row.get("promise_premium_focus")
    urgency = row.get("promise_urgency_focus")
    vals = []
    for v, w in zip([discount, premium, urgency], PROMISE_PRESSURE_WEIGHTS):
        if pd.notna(v) and isinstance(v, (int, float)):
            vals.append(float(v) * w)
    if not vals:
        return 0.0
    return float(np.clip(sum(vals), 0.0, 1.0))


def _expectation_score_from_creative(creative_id: str, creatives_df: pd.DataFrame) -> float:
    """
    Expectation score from creative promise axes. Uses all 8 when present
    (transformation/status weighted higher).  Tier 9: angle expectation_lift
    shifts the baseline upward for high-expectation angles.
    """
    if creatives_df.empty or creative_id is None or (isinstance(creative_id, float) and np.isnan(creative_id)):
        return 0.5
    row = creatives_df[creatives_df["creative_id"] == creative_id]
    if row.empty:
        return 0.5
    row = row.iloc[0]
    have_8 = any(ax in row.index and pd.notna(row.get(ax)) for ax in PROMISE_8_AXES)
    base_expectation = 0.5
    if have_8:
        total, wsum = 0.0, 0.0
        for ax in PROMISE_8_AXES:
            if ax not in row.index:
                continue
            v = row[ax]
            if pd.notna(v) and isinstance(v, (int, float)):
                w = EXPECTATION_8_WEIGHTS.get(ax, 1.0)
                total += float(v) * w
                wsum += w
        if wsum > 0:
            base_expectation = float(np.clip(total / wsum, 0.0, 1.0))
    else:
        vals = []
        for ax in PROMISE_AXES:
            if ax in row.index:
                v = row[ax]
                if pd.notna(v) and isinstance(v, (int, float)):
                    vals.append(float(v))
        if vals:
            base_expectation = float(np.clip(np.mean(vals), 0.0, 1.0))

    # Tier 9: angle expectation lift
    angle = row.get("creative_angle") if "creative_angle" in row.index else None
    if angle is not None and pd.notna(angle):
        angle_params, _, _ = _get_tier9_params()
        if angle_params and angle in angle_params:
            base_expectation += angle_params[angle].get("expectation_lift", 0.0)

    return float(np.clip(base_expectation, 0.0, 1.0))


def _trait_creative_alignment(
    customer_row: pd.Series, creative_id, creatives_df: pd.DataFrame
) -> float:
    """Tier 5: alignment between customer traits and creative promises. Uses 8 axes when present."""
    if creatives_df.empty or creative_id is None or (isinstance(creative_id, float) and np.isnan(creative_id)):
        return 0.5
    row = creatives_df[creatives_df["creative_id"] == creative_id]
    if row.empty:
        return 0.5
    row = row.iloc[0]
    matches = []
    match_discount = customer_row.get("price_sensitivity", 0.5) * row.get("promise_discount_focus", 0.5)
    match_urgency = customer_row.get("impulse_level", 0.5) * row.get("promise_urgency_focus", 0.5)
    match_premium = customer_row.get("quality_expectation", 0.5) * row.get("promise_premium_focus", 0.5)
    match_safety = (1.0 - customer_row.get("regret_propensity", 0.5)) * row.get("promise_safety_focus", 0.5)
    matches = [match_discount, match_urgency, match_premium, match_safety]
    have_8 = any(ax in row.index and pd.notna(row.get(ax)) for ax in PROMISE_8_AXES)
    if have_8:
        impulse = customer_row.get("impulse_level", 0.5)
        quality = customer_row.get("quality_expectation", 0.5)
        regret = customer_row.get("regret_propensity", 0.5)
        loyalty = customer_row.get("loyalty_propensity", 0.5)
        if "promise_status_intensity" in row.index and pd.notna(row.get("promise_status_intensity")):
            matches.append(impulse * row["promise_status_intensity"])
        if "promise_belonging_intensity" in row.index and pd.notna(row.get("promise_belonging_intensity")):
            matches.append((1.0 - regret) * row["promise_belonging_intensity"])
        if "promise_control_intensity" in row.index and pd.notna(row.get("promise_control_intensity")):
            matches.append((1.0 - regret) * row["promise_control_intensity"])
        if "promise_transformation_intensity" in row.index and pd.notna(row.get("promise_transformation_intensity")):
            matches.append(quality * row["promise_transformation_intensity"])
        if "promise_relief_intensity" in row.index and pd.notna(row.get("promise_relief_intensity")):
            matches.append(impulse * row["promise_relief_intensity"])
        if "promise_novelty_intensity" in row.index and pd.notna(row.get("promise_novelty_intensity")):
            matches.append(impulse * row["promise_novelty_intensity"])
        if "promise_mastery_intensity" in row.index and pd.notna(row.get("promise_mastery_intensity")):
            matches.append(loyalty * row["promise_mastery_intensity"])
        if "promise_safety_intensity" in row.index and pd.notna(row.get("promise_safety_intensity")):
            matches.append((1.0 - regret) * row["promise_safety_intensity"])
    return float(np.clip(np.mean(matches), 0.0, 1.0))


def _experience_score_for_order(
    order_id: str,
    order_created_at: pd.Timestamp,
    line_items_df: pd.DataFrame,
    products_df: pd.DataFrame,
    fulfillments_df: pd.DataFrame,
    refunded_order_ids: set,
) -> float:
    """
    Experience score from order: quality, shipping, refund.
    High quality + fast shipping + no refund → high; low quality or refund → low.
    Returns value in [0, 1].
    """
    if order_id in refunded_order_ids:
        return 0.15

    # Quality: mean of product quality_level in this order
    li = line_items_df[line_items_df["order_id"] == order_id]
    if li.empty:
        quality_avg = 0.5
    else:
        li = li.merge(products_df[["product_id", "quality_level"]], on="product_id", how="left")
        li["_q"] = li["quality_level"].map(_QUALITY_TO_SCORE).fillna(0.5)
        quality_avg = float(li["_q"].mean())

    # Shipping: days from order to delivery (if we have fulfillment)
    fast_bonus = 0.0
    if not fulfillments_df.empty and "order_id" in fulfillments_df.columns:
        ful = fulfillments_df[fulfillments_df["order_id"] == order_id]
        if not ful.empty and "delivered_at" in ful.columns:
            delivered_at = pd.to_datetime(ful.iloc[0]["delivered_at"])
            order_ts = pd.to_datetime(order_created_at)
            days = (delivered_at - order_ts).days
            if days <= 5:
                fast_bonus = 1.0
            elif days <= 10:
                fast_bonus = 0.5

    # Base 0.3 + quality up to 0.5 + shipping bonus up to 0.2
    score = 0.3 + 0.5 * quality_avg + 0.2 * fast_bonus
    return float(np.clip(score, 0.0, 1.0))


def update_customers_from_day_orders(
    customers_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    line_items_df: pd.DataFrame,
    products_df: pd.DataFrame,
    fulfillments_df: pd.DataFrame,
    refunds_df: pd.DataFrame,
    creatives_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    After the day's orders, refunds, and fulfillments: compute expectation vs experience
    per order, then update each customer's trust_score, disappointment_memory, satisfaction_memory.
    State persists; future days use the updated customers_df.

    Config: trust_alpha (disappointment → trust decrease), trust_beta (satisfaction → trust increase).
    Defaults: alpha=0.1, beta=0.1.
    """
    alpha = float(config.get("trust_alpha", 0.1))
    beta = float(config.get("trust_beta", 0.1))
    velocity_decay = float(config.get("negative_velocity_decay", 0.85))
    velocity_cap = float(config.get("negative_velocity_cap", 2.0))

    if orders_df.empty:
        return customers_df.copy()

    # Ensure state columns exist (e.g. if customers came from a run without them)
    for col in ["trust_score", "disappointment_memory", "satisfaction_memory"]:
        if col not in customers_df.columns:
            customers_df = customers_df.copy()
            customers_df[col] = 0.5 if col == "trust_score" else 0.0
    if "recent_negative_velocity" not in customers_df.columns:
        customers_df = customers_df.copy()
        customers_df["recent_negative_velocity"] = 0.0

    customers_df = customers_df.copy()
    cid_to_idx = {cid: i for i, cid in enumerate(customers_df["customer_id"].astype(str))}
    trust = customers_df["trust_score"].values.copy()
    disappointment = customers_df["disappointment_memory"].values.copy()
    satisfaction = customers_df["satisfaction_memory"].values.copy()
    recent_neg_vel = customers_df["recent_negative_velocity"].fillna(0).values.copy()
    if "price_sensitivity" in customers_df.columns:
        price_sensitivity = customers_df["price_sensitivity"].values.copy()
    else:
        price_sensitivity = None

    discount_drift_strength = float(config.get("discount_drift_strength", 0.01))
    refunded_order_ids = set(refunds_df["order_id"].astype(str)) if not refunds_df.empty else set()
    created_col = "created_at" if "created_at" in orders_df.columns else "order_timestamp"
    refund_penalty_indices = set()  # Track customers with refunded orders for post-EMA penalty

    for _, order in orders_df.iterrows():
        oid = order.get("id")
        cid = order.get("customer_id")
        if pd.isna(oid) or oid is None or pd.isna(cid) or cid is None:
            continue
        cid_str = str(cid)
        if cid_str not in cid_to_idx:
            continue
        idx = cid_to_idx[cid_str]
        customer_row = customers_df.iloc[idx]

        creative_id = order.get("last_attributed_creative_id")
        expectation = _expectation_score_from_creative(creative_id, creatives_df)
        order_created = order.get(created_col) if created_col else pd.Timestamp.now()
        experience = _experience_score_for_order(
            str(oid),
            pd.to_datetime(order_created),
            line_items_df,
            products_df,
            fulfillments_df,
            refunded_order_ids,
        )
        # Track refunded orders for post-EMA direct penalty
        if str(oid) in refunded_order_ids:
            refund_penalty_indices.add(idx)
        mismatch = expectation - experience

        # Price psychology: discount shifts mismatch (premium-seeking suspicious, price-sensitive relieved)
        discount_pct = order.get("discount_pct", 0) or 0
        if pd.notna(discount_pct) and float(discount_pct) > 0:
            qe = customer_row.get("quality_expectation", 0.5)
            ps = customer_row.get("price_sensitivity", 0.5)
            price_mismatch_adjustment = float(discount_pct) * (float(qe) - float(ps))
            mismatch = mismatch + price_mismatch_adjustment
            if price_sensitivity is not None:
                price_sensitivity[idx] = np.clip(
                    price_sensitivity[idx] + float(discount_pct) * discount_drift_strength, 0.0, 1.0
                )

        # Tier 5: promise pressure amplifies disappointment (high-pressure creatives decay trust faster)
        promise_pressure = _promise_pressure_from_creative(creative_id, creatives_df)
        alignment = _trait_creative_alignment(customer_row, creative_id, creatives_df)

        # Tier 9: angle trust stability modifier
        angle_trust_mod = 1.0
        creative_row = creatives_df[creatives_df["creative_id"] == creative_id] if creative_id is not None else pd.DataFrame()
        if not creative_row.empty:
            cr = creative_row.iloc[0]
            angle = cr.get("creative_angle") if "creative_angle" in cr.index else None
            if angle is not None and pd.notna(angle):
                angle_params, _, ct_params = _get_tier9_params()
                if angle_params and angle in angle_params:
                    angle_trust_mod = angle_params[angle].get("trust_stability", 1.0)
            # Tier 9: creative type trust baseline shift (small)
            ct = cr.get("creative_type") if "creative_type" in cr.index else None
            if ct is not None and pd.notna(ct):
                _, _, ct_params = _get_tier9_params()
                if ct_params and ct in ct_params:
                    trust[idx] += ct_params[ct].get("trust_baseline_shift", 0.0) * 0.1

        if mismatch > 0:
            effective_mismatch = mismatch * (1.0 + promise_pressure)
            effective_mismatch *= (1.0 + (1.0 - alignment))
            # Upgrade 6: positive history tolerance — 3+ good experiences dampen disappointment
            if "shipping_good_count" in customers_df.columns:
                good_count = int(customers_df.iloc[idx].get("shipping_good_count", 0) or 0)
                if good_count >= 3:
                    effective_mismatch *= 0.7
            # Tier 9: angle trust stability — authority slows decay; fear sharpens it
            effective_alpha = alpha / angle_trust_mod
            disappointment[idx] += effective_mismatch
            trust[idx] -= effective_mismatch * effective_alpha
            recent_neg_vel[idx] = velocity_decay * recent_neg_vel[idx] + (1.0 - velocity_decay) * min(mismatch, 1.0)
        else:
            satisfaction_bonus = abs(mismatch) * (1.0 + alignment)
            # Tier 9: angle trust stability — authority compounds satisfaction faster
            effective_beta = beta * angle_trust_mod
            satisfaction[idx] += satisfaction_bonus
            trust[idx] += satisfaction_bonus * effective_beta
            recent_neg_vel[idx] = velocity_decay * recent_neg_vel[idx]

    trust = np.clip(trust, 0.0, 1.0)
    disappointment = np.maximum(disappointment, 0.0)
    satisfaction = np.maximum(satisfaction, 0.0)
    recent_neg_vel = np.minimum(recent_neg_vel, velocity_cap)
    recent_neg_vel = np.maximum(recent_neg_vel, 0.0)

    # Upgrade 3: Trust EMA buffering — smooth trust changes over time
    trust_ema_weight = float(config.get("lag_structure", {}).get("trust_ema_weight", 1.0) if isinstance(config.get("lag_structure"), dict) else 1.0)
    if trust_ema_weight < 1.0:
        old_trust = customers_df["trust_score"].values
        trust = trust_ema_weight * old_trust + (1.0 - trust_ema_weight) * trust
        trust = np.clip(trust, 0.0, 1.0)

    # Post-EMA direct refund trust penalty (bypasses EMA dampening)
    refund_direct_penalty = float(config.get("feedback_loops", {}).get("refund_direct_trust_penalty", 0.08))
    if refund_direct_penalty > 0 and refund_penalty_indices:
        for idx in refund_penalty_indices:
            trust[idx] -= refund_direct_penalty
        trust = np.clip(trust, 0.0, 1.0)

    customers_df["trust_score"] = trust
    customers_df["disappointment_memory"] = disappointment
    customers_df["satisfaction_memory"] = satisfaction
    customers_df["recent_negative_velocity"] = recent_neg_vel
    if price_sensitivity is not None:
        customers_df["price_sensitivity"] = np.clip(price_sensitivity, 0.0, 1.0)
    return customers_df


def apply_identity_drift(
    customers_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    creatives_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Tier 4 identity drift: each exposure nudges traits toward creative promises.
    Tier 5: high-pressure creatives add toxic growth (quality_expectation + regret_propensity);
    drift strength scales with exposure history (accumulation).
    Tier 9: persona influences drift direction; value proposition amplifies specific traits.
    """
    strength = float(config.get("identity_drift_strength", 0.01))
    toxic_strength = float(config.get("drift_toxic_pressure_strength", 0.015))
    accum_factor = float(config.get("drift_accumulation_factor", 0.02))
    accum_cap = float(config.get("drift_accumulation_cap", 50.0))

    if exposure_df.empty or creatives_df.empty:
        return customers_df
    needed = ["customer_id", "creative_id"]
    if not all(c in exposure_df.columns for c in needed):
        return customers_df
    promise_cols = [c for c in PROMISE_AXES if c in creatives_df.columns]
    for ax in PROMISE_8_AXES:
        if ax in creatives_df.columns and ax not in promise_cols:
            promise_cols.append(ax)
    if not promise_cols:
        return customers_df

    # Merge Tier 9 creative strategy columns along with promises
    merge_cols = ["creative_id"] + promise_cols
    for t9col in ["creative_persona", "creative_value_proposition"]:
        if t9col in creatives_df.columns and t9col not in merge_cols:
            merge_cols.append(t9col)

    exp = exposure_df[["customer_id", "creative_id"]].merge(
        creatives_df[merge_cols],
        on="creative_id",
        how="left",
    )
    for c in promise_cols:
        exp[c] = exp[c].fillna(0).clip(0, 1)

    exp["_pressure"] = exp["creative_id"].map(
        lambda cid: _promise_pressure_from_creative(cid, creatives_df)
    )

    # 4-axis + 8-axis identity drift: small daily nudges (no labels)
    axis_to_trait = [
        ("promise_discount_focus", "price_sensitivity", 1),
        ("promise_premium_focus", "quality_expectation", 1),
        ("promise_urgency_focus", "impulse_level", 1),
        ("promise_safety_focus", "regret_propensity", -1),
        ("promise_belonging_intensity", "loyalty_propensity", 1),
        ("promise_transformation_intensity", "quality_expectation", 1),
        ("promise_novelty_intensity", "impulse_level", 1),
        ("promise_status_intensity", "impulse_level", 1),
        ("promise_mastery_intensity", "loyalty_propensity", 1),
        ("promise_control_intensity", "regret_propensity", -1),
        ("promise_safety_intensity", "regret_propensity", -1),
    ]
    drift_scale_8 = float(config.get("identity_drift_8_scale", 0.6))
    trait_cols_unique = list(dict.fromkeys([tcol for _pcol, tcol, _sign in axis_to_trait if tcol in customers_df.columns]))
    trait_deltas = {tcol: np.zeros(len(exp)) for tcol in trait_cols_unique}
    for pcol, tcol, sign in axis_to_trait:
        if pcol not in exp.columns or tcol not in customers_df.columns:
            continue
        mult = drift_scale_8 if pcol in PROMISE_8_AXES else 1.0
        trait_deltas[tcol] = trait_deltas[tcol] + exp[pcol].values * strength * sign * mult
    for tcol, arr in trait_deltas.items():
        exp["_d_" + tcol] = arr
    if "quality_expectation" in trait_deltas and "_pressure" in exp.columns:
        exp["_d_quality_expectation"] = exp["_d_quality_expectation"] + exp["_pressure"] * toxic_strength
    if "regret_propensity" in trait_deltas and "_pressure" in exp.columns:
        exp["_d_regret_propensity"] = exp["_d_regret_propensity"] + exp["_pressure"] * toxic_strength

    # Tier 9: persona-driven drift direction — personas nudge traits toward their affinity centers
    _, persona_affinity, _ = _get_tier9_params()
    persona_drift_strength = float(config.get("persona_drift_strength", 0.004))
    if persona_affinity and "creative_persona" in exp.columns:
        for trait in trait_cols_unique:
            persona_nudge = np.zeros(len(exp))
            for i in range(len(exp)):
                persona = exp["creative_persona"].iat[i]
                if persona is not None and pd.notna(persona) and persona in persona_affinity:
                    aff = persona_affinity[persona]
                    if trait in aff:
                        center, _ = aff[trait]
                        persona_nudge[i] = (center - 0.5) * persona_drift_strength
            exp["_d_" + trait] = exp["_d_" + trait].values + persona_nudge

    # Tier 9: value proposition trait amplification — VP biases which traits grow faster
    from .meta import _VP_PROMISE_BIAS
    vp_drift_strength = float(config.get("vp_drift_strength", 0.003))
    _p8_to_trait_drift = {
        "promise_transformation_intensity": "quality_expectation",
        "promise_status_intensity": "impulse_level",
        "promise_belonging_intensity": "loyalty_propensity",
        "promise_safety_intensity": "regret_propensity",
        "promise_control_intensity": "regret_propensity",
        "promise_novelty_intensity": "impulse_level",
        "promise_mastery_intensity": "loyalty_propensity",
        "promise_relief_intensity": "impulse_level",
    }
    _p8_to_sign = {
        "promise_safety_intensity": -1, "promise_control_intensity": -1,
    }
    if "creative_value_proposition" in exp.columns:
        for trait in trait_cols_unique:
            vp_nudge = np.zeros(len(exp))
            for i in range(len(exp)):
                vp = exp["creative_value_proposition"].iat[i]
                if vp is not None and pd.notna(vp) and vp in _VP_PROMISE_BIAS:
                    biases = _VP_PROMISE_BIAS[vp]
                    for p8_axis, (mean_shift, _) in biases.items():
                        mapped_trait = _p8_to_trait_drift.get(p8_axis)
                        if mapped_trait == trait:
                            sign = _p8_to_sign.get(p8_axis, 1)
                            vp_nudge[i] += mean_shift * vp_drift_strength * sign
            exp["_d_" + trait] = exp["_d_" + trait].values + vp_nudge

    dcol_to_tcol = {"_d_" + tcol: tcol for tcol in trait_deltas}
    agg = exp.groupby("customer_id", as_index=False).agg({d: "sum" for d in dcol_to_tcol})
    agg = agg.rename(columns={d: "delta_" + t for d, t in dcol_to_tcol.items()})

    customers_df = customers_df.copy()
    use_accum = "exposure_count" in customers_df.columns and accum_factor > 0
    if use_accum:
        exposure_count = customers_df["exposure_count"].fillna(0).values
        accum_mult = 1.0 + np.minimum(exposure_count, accum_cap) * accum_factor
    else:
        accum_mult = np.ones(len(customers_df))

    for tcol in trait_cols_unique:
        delta_col = "delta_" + tcol
        if delta_col not in agg.columns or tcol not in customers_df.columns:
            continue
        merged = customers_df[["customer_id"]].merge(
            agg[["customer_id", delta_col]], on="customer_id", how="left"
        )
        merged[delta_col] = merged[delta_col].fillna(0)
        delta_vals = merged[delta_col].values * accum_mult
        customers_df[tcol] = customers_df[tcol].values + delta_vals
        customers_df[tcol] = np.clip(customers_df[tcol].values, 0.0, 1.0)
    return customers_df


def update_shipping_experience(
    customers_df: pd.DataFrame,
    fulfillments_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    current_date,
) -> pd.DataFrame:
    """
    Upgrade 6: Track shipping experience from today's fulfillments.
    Fast delivery (<=5 days) → shipping_good_count++
    Slow delivery (>10 days) → shipping_bad_count++
    """
    if fulfillments_df.empty or orders_df.empty:
        return customers_df
    if "shipping_bad_count" not in customers_df.columns:
        return customers_df

    # Join fulfillments to orders to get customer_id and order date
    ful = fulfillments_df.copy()
    if "order_id" not in ful.columns:
        return customers_df
    ful = ful.merge(
        orders_df[["id", "customer_id", "created_at"]].rename(columns={"id": "order_id"}),
        on="order_id", how="left",
    )
    if "delivered_at" not in ful.columns or ful.empty:
        return customers_df

    ful["_delivery_days"] = (
        pd.to_datetime(ful["delivered_at"]) - pd.to_datetime(ful["created_at"])
    ).dt.days

    customers_df = customers_df.copy()
    cid_to_idx = {str(cid): i for i, cid in enumerate(customers_df["customer_id"].values)}
    bad = customers_df["shipping_bad_count"].values.copy()
    good = customers_df["shipping_good_count"].values.copy()

    for _, row in ful.iterrows():
        cid = str(row.get("customer_id", ""))
        if cid not in cid_to_idx:
            continue
        idx = cid_to_idx[cid]
        days = row.get("_delivery_days")
        if pd.isna(days):
            continue
        days = int(days)
        if days <= 5:
            good[idx] += 1
        elif days > 10:
            bad[idx] += 1

    customers_df["shipping_bad_count"] = bad
    customers_df["shipping_good_count"] = good
    return customers_df


def apply_brand_memory_decay(
    customers_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Upgrade 6: Inactive customers (days_since_last_interaction > threshold)
    have trust drift toward 0.5 (brand forgotten). Applied once per day.
    """
    mem = config.get("memory", {})
    decay_after = int(mem.get("brand_memory_decay_after_days", 90))

    if "days_since_last_interaction" not in customers_df.columns:
        return customers_df
    if "trust_score" not in customers_df.columns:
        return customers_df

    customers_df = customers_df.copy()
    days_inactive = customers_df["days_since_last_interaction"].fillna(0).values
    trust = customers_df["trust_score"].values.copy()

    # For customers inactive > threshold, drift trust toward 0.5
    inactive_mask = days_inactive > decay_after
    if not np.any(inactive_mask):
        return customers_df

    # Drift rate: small daily pull toward 0.5
    decay_rate = float(mem.get("brand_memory_decay_rate", 0.005))
    trust[inactive_mask] += (0.5 - trust[inactive_mask]) * decay_rate

    customers_df["trust_score"] = np.clip(trust, 0.0, 1.0)
    return customers_df
