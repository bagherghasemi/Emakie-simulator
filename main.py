"""
Multi-day behavioral simulation.

Setup (once): load config, generate static entities (customers, products,
variants, creatives, campaigns, adsets, ads), publish them to BigQuery when
project_id/dataset_meta/dataset_shopify are configured. Then iterate day by day
from config.simulation.start_date to end_date: simulate_daily_exposure → extract
clicks → simulate_purchases_from_clicks → simulate_refunds. Write daily parquet
files to output/ subfolders; no in-memory accumulation.
"""

# Allow running as a script: python main.py (no parent package on path)
if __package__ is None or __package__ == "":
    import os
    import sys
    import types
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    # So that "from emakie_simulator.loaders import ..." resolves when run from repo root
    _pkg = types.ModuleType("emakie_simulator")
    _pkg.__path__ = [_script_dir]
    sys.modules["emakie_simulator"] = _pkg
    # So that "from generators...", "from loaders..." resolve
    sys.path.insert(0, _script_dir)

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from google.cloud import bigquery

from generators.aftermath import simulate_refunds
from generators.brand_state import BrandState
from generators.momentum import MomentumTracker
from generators.population import PopulationManager
from generators.psychological_state import (
    update_customers_from_day_orders,
    apply_identity_drift,
    update_shipping_experience,
    apply_brand_memory_decay,
)
from generators.commerce import (
    generate_products,
    generate_variants,
    simulate_cart_returns,
    simulate_purchases_from_clicks,
    simulate_repeat_purchases_for_day,
)
from generators.humans import generate_customers, generate_prospects, get_empty_customers_schema
from generators.meta import (
    generate_ad_accounts,
    generate_ads,
    generate_adsets,
    generate_campaigns,
    generate_creatives,
    simulate_daily_exposure,
    update_creative_lifecycle,
)
from generators.meta_reporting import build_ad_performance_daily
from generators.operations import simulate_fulfillments
from loaders.static_entities import load_static_entities

OUTPUT_ROOT = Path(__file__).parent / "output"


def _assert_shopify_static_hierarchy(
    products_df: pd.DataFrame, variants_df: pd.DataFrame
) -> None:
    """product_variants.product_id → products.id. Fail fast if broken."""
    if variants_df.empty:
        return
    valid_product_ids = set(products_df["id"].astype(str))
    refs = set(variants_df["product_id"].astype(str))
    bad = refs - valid_product_ids
    if bad:
        raise ValueError(
            "Shopify hierarchy broken: variant.product_id must reference products.id. "
            f"Found {len(bad)} variant(s) with product_id not in products: {list(bad)[:5]}..."
        )


def _assert_shopify_orders_hierarchy(
    orders_df: pd.DataFrame,
    line_items_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    variants_df: pd.DataFrame,
) -> None:
    """Assert: orders.customer_id→customers.id; line_items.order_id→orders.id; line_items.variant_id→variants.id; line_items.product_id→products.id; transactions.order_id→orders.id."""
    if orders_df.empty:
        return
    order_ids = set(orders_df["id"].astype(str))
    customer_ids = set(customers_df["id"].astype(str))
    variant_ids = set(variants_df["id"].astype(str))
    product_ids = set(products_df["id"].astype(str))

    bad_customer = set(orders_df["customer_id"].astype(str)) - customer_ids
    if bad_customer:
        raise ValueError(
            "Shopify hierarchy broken: orders.customer_id must reference customers.id. "
            f"Found {len(bad_customer)} order(s) with customer_id not in customers."
        )

    if not line_items_df.empty:
        bad_order_li = set(line_items_df["order_id"].astype(str)) - order_ids
        if bad_order_li:
            raise ValueError(
                "Shopify hierarchy broken: order_line_items.order_id must reference orders.id. "
                f"Found {len(bad_order_li)} line_item(s) with order_id not in orders."
            )
        bad_variant = set(line_items_df["variant_id"].astype(str)) - variant_ids
        if bad_variant:
            raise ValueError(
                "Shopify hierarchy broken: order_line_items.variant_id must reference product_variants.id. "
                f"Found {len(bad_variant)} line_item(s) with variant_id not in variants."
            )
        bad_product_li = set(line_items_df["product_id"].astype(str)) - product_ids
        if bad_product_li:
            raise ValueError(
                "Shopify hierarchy broken: order_line_items.product_id must reference products.id. "
                f"Found {len(bad_product_li)} line_item(s) with product_id not in products."
            )

    if not transactions_df.empty:
        bad_order_tx = set(transactions_df["order_id"].astype(str)) - order_ids
        if bad_order_tx:
            raise ValueError(
                "Shopify hierarchy broken: transactions.order_id must reference orders.id. "
                f"Found {len(bad_order_tx)} transaction(s) with order_id not in orders."
            )


def _assert_shopify_refunds_hierarchy(
    refunds_df: pd.DataFrame, orders_df: pd.DataFrame
) -> None:
    """refunds.order_id → orders.id."""
    if refunds_df.empty:
        return
    order_ids = set(orders_df["id"].astype(str))
    bad = set(refunds_df["order_id"].astype(str)) - order_ids
    if bad:
        raise ValueError(
            "Shopify hierarchy broken: refunds.order_id must reference orders.id. "
            f"Found {len(bad)} refund(s) with order_id not in orders."
        )


def _assert_shopify_fulfillments_hierarchy(
    fulfillments_df: pd.DataFrame, orders_df: pd.DataFrame
) -> None:
    """fulfillments.order_id → orders.id."""
    if fulfillments_df.empty:
        return
    order_ids = set(orders_df["id"].astype(str))
    bad = set(fulfillments_df["order_id"].astype(str)) - order_ids
    if bad:
        raise ValueError(
            "Shopify hierarchy broken: fulfillments.order_id must reference orders.id. "
            f"Found {len(bad)} fulfillment(s) with order_id not in orders."
        )


def _enrich_adsets_with_account_id(
    adsets_df: pd.DataFrame, campaigns_df: pd.DataFrame
) -> pd.DataFrame:
    """Add account_id to adsets by joining campaigns on campaign_id. Overwrites if present."""
    camp = campaigns_df[["id", "account_id"]].rename(columns={"id": "_campaign_id_key"})
    out = adsets_df.drop(columns=["account_id"], errors="ignore").merge(
        camp, left_on="campaign_id", right_on="_campaign_id_key", how="left"
    )
    return out.drop(columns=["_campaign_id_key"])


def _enrich_ads_with_campaign_and_account(
    ads_df: pd.DataFrame, adsets_df: pd.DataFrame
) -> pd.DataFrame:
    """Add campaign_id and account_id to ads by joining adsets on adset_id.
    Preserves creative_id from ads_df (never dropped or overwritten)."""
    required = ["id", "adset_id", "creative_id", "campaign_id", "account_id"]
    aset = adsets_df[["id", "campaign_id", "account_id"]].rename(
        columns={"id": "_adset_id_key"}
    )
    out = ads_df.drop(columns=["campaign_id", "account_id"], errors="ignore").merge(
        aset, left_on="adset_id", right_on="_adset_id_key", how="left"
    )
    out = out.drop(columns=["_adset_id_key"])
    _require_ads_hierarchy(out)
    # Guarantee column order: required first, then rest
    rest = [c for c in out.columns if c not in required]
    return out[required + rest]


def _enrich_ads_with_flight_dates(
    ads_df: pd.DataFrame, adsets_df: pd.DataFrame
) -> pd.DataFrame:
    """Set start_time and end_time on each ad from its ad set (parent constrains child)."""
    if "id" not in adsets_df.columns or "start_time" not in adsets_df.columns or "end_time" not in adsets_df.columns:
        return ads_df
    lookup = adsets_df[["id", "start_time", "end_time"]].rename(columns={"id": "_adset_key"})
    out = ads_df.drop(columns=["start_time", "end_time"], errors="ignore").merge(
        lookup, left_on="adset_id", right_on="_adset_key", how="left"
    ).drop(columns=["_adset_key"])
    return out


def _enrich_ads_with_audience_type_and_objective(
    ads_df: pd.DataFrame, adsets_df: pd.DataFrame, campaigns_df: pd.DataFrame,
) -> pd.DataFrame:
    """Tier 10: propagate audience_type (from adset) and objective (from campaign) onto ads.

    These flow through the real hierarchy: campaign→adset→ad. No duplication logic —
    ads look up audience_type via adset_id and objective via campaign_id.
    """
    out = ads_df.copy()
    if "audience_type" in adsets_df.columns:
        aset_lookup = adsets_df[["id", "audience_type"]].rename(columns={"id": "_aset_key"})
        out = out.drop(columns=["audience_type"], errors="ignore").merge(
            aset_lookup, left_on="adset_id", right_on="_aset_key", how="left",
        ).drop(columns=["_aset_key"])
    if "objective" in campaigns_df.columns:
        camp_lookup = campaigns_df[["id", "objective"]].rename(columns={"id": "_camp_key"})
        out = out.drop(columns=["objective"], errors="ignore").merge(
            camp_lookup, left_on="campaign_id", right_on="_camp_key", how="left",
        ).drop(columns=["_camp_key"])
    return out


def _enrich_ads_with_creative_name_and_type(
    ads_df: pd.DataFrame, creatives_df: pd.DataFrame
) -> pd.DataFrame:
    """Set creative_name and creative_type on each ad from meta_creatives (join on creative_id)."""
    if "creative_id" not in creatives_df.columns:
        return ads_df
    lookup_cols = ["creative_id"]
    if "creative_name" in creatives_df.columns:
        lookup_cols.append("creative_name")
    if "creative_type" in creatives_df.columns:
        lookup_cols.append("creative_type")
    if len(lookup_cols) == 1:
        return ads_df
    lookup = creatives_df[lookup_cols].drop_duplicates("creative_id")
    drop_cols = [c for c in ["creative_name", "creative_type"] if c in ads_df.columns]
    out = ads_df.drop(columns=drop_cols, errors="ignore").merge(
        lookup, on="creative_id", how="left"
    )
    required = ["id", "adset_id", "creative_id", "campaign_id", "account_id"]
    extra = [c for c in ["creative_name", "creative_type"] if c in out.columns]
    rest = [c for c in out.columns if c not in required and c not in extra]
    return out[required + extra + rest]


def _require_ads_hierarchy(ads_df: pd.DataFrame) -> None:
    """Raise if ads_df has any NULL creative_id or missing required hierarchy columns."""
    required = ["id", "adset_id", "creative_id", "campaign_id", "account_id"]
    for col in required:
        if col not in ads_df.columns:
            raise ValueError(f"meta_ads must have column '{col}'")
    null_creative = ads_df["creative_id"].isna()
    if null_creative.any():
        n = null_creative.sum()
        raise ValueError(
            f"meta_ads must not have NULL creative_id (generate_ads links every ad to a creative). Found {n} row(s) with NULL creative_id."
        )


def _apply_first_touch_attribution(
    customers_df: pd.DataFrame, exposure_history_dfs: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    Set first-touch attribution on customers: earliest click, else earliest impression.
    Priority: 1) earliest click, 2) else earliest impression. Hierarchy from ads.
    Stable: a customer's first touch is never changed after assignment.
    """
    first_touch_cols = [
        "first_touch_ad_id",
        "first_touch_adset_id",
        "first_touch_campaign_id",
        "first_touch_creative_id",
        "first_touch_date",
    ]
    empty_first_touch = pd.DataFrame(columns=["customer_id"] + first_touch_cols)

    if not exposure_history_dfs:
        for c in first_touch_cols:
            customers_df[c] = pd.NA
        return customers_df

    all_exposures = pd.concat(exposure_history_dfs, ignore_index=True)
    if all_exposures.empty:
        customers_df = customers_df.copy()
        for c in first_touch_cols:
            if c not in customers_df.columns:
                customers_df[c] = pd.NA
        return customers_df

    # Earliest click first, else earliest impression.
    # Tie-breaks must be deterministic (within-day ordering + stable sort).
    sort_cols = ["clicked"]
    asc = [False]
    if "exposure_timestamp" in all_exposures.columns:
        sort_cols.append("exposure_timestamp")
        asc.append(True)
    else:
        sort_cols.append("date")
        asc.append(True)
        if "multi_ad_session_depth" in all_exposures.columns:
            sort_cols.append("multi_ad_session_depth")
            asc.append(True)
    # Final deterministic tie-breakers
    for c in ["ad_id", "creative_id"]:
        if c in all_exposures.columns:
            sort_cols.append(c)
            asc.append(True)
    all_exposures = all_exposures.sort_values(by=sort_cols, ascending=asc, kind="mergesort")
    first = all_exposures.groupby("customer_id", as_index=False, sort=False).first()

    rename_map = {
        "date": "first_touch_date",
        "ad_id": "first_touch_ad_id",
        "creative_id": "first_touch_creative_id",
    }
    if "adset_id" in first.columns:
        rename_map["adset_id"] = "first_touch_adset_id"
    if "campaign_id" in first.columns:
        rename_map["campaign_id"] = "first_touch_campaign_id"

    first_touch_df = first.rename(columns=rename_map)
    out_cols = ["customer_id", "first_touch_date", "first_touch_ad_id", "first_touch_adset_id", "first_touch_campaign_id", "first_touch_creative_id"]
    out_cols = [c for c in out_cols if c in first_touch_df.columns]
    first_touch_df = first_touch_df[out_cols]

    for c in first_touch_cols:
        if c not in first_touch_df.columns:
            first_touch_df[c] = pd.NA

    # Apply to customers without overwriting existing non-null values (no backfill/override).
    customers_df = customers_df.copy()
    for c in first_touch_cols:
        if c not in customers_df.columns:
            customers_df[c] = pd.NA

    ft = customers_df[["customer_id"]].merge(first_touch_df, on="customer_id", how="left")
    for c in first_touch_cols:
        if c in ft.columns:
            customers_df[c] = customers_df[c].combine_first(ft[c])
    return customers_df


def _update_fatigue_after_exposure(
    customers_df: pd.DataFrame, exposure_df: pd.DataFrame, config: dict,
    creatives_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Tier 4 creative fatigue: after daily exposure, increment per (customer, creative),
    then apply decay to all entries and prune small values.
    Tier 9: hook fatigue acceleration and creative type fatigue curve modify the
    increment per exposure so some creatives fatigue faster than others.
    """
    if "creative_fatigue_map" not in customers_df.columns or exposure_df.empty:
        return customers_df
    decay = float(config.get("fatigue_decay", 0.97))
    thresh = 0.01

    # Tier 9: pre-compute per-creative fatigue multiplier from hook + creative_type
    from generators.meta import _HOOK_PARAMS, _CREATIVE_TYPE_PARAMS, _ANGLE_PARAMS
    creative_fatigue_mult: dict[str, float] = {}
    if creatives_df is not None and not creatives_df.empty:
        for _, cr in creatives_df.iterrows():
            crid = str(cr["creative_id"])
            mult = 1.0
            hook = cr.get("creative_hook_pattern")
            if hook is not None and pd.notna(hook) and hook in _HOOK_PARAMS:
                mult *= _HOOK_PARAMS[hook].get("fatigue_accel", 1.0)
            ct = cr.get("creative_type")
            if ct is not None and pd.notna(ct) and ct in _CREATIVE_TYPE_PARAMS:
                mult *= _CREATIVE_TYPE_PARAMS[ct].get("fatigue_curve", 1.0)
            angle = cr.get("creative_angle")
            if angle is not None and pd.notna(angle) and angle in _ANGLE_PARAMS:
                mult *= _ANGLE_PARAMS[angle].get("fatigue_rate", 1.0)
            creative_fatigue_mult[crid] = mult

    customers_df = customers_df.copy()
    fatigue_col = customers_df["creative_fatigue_map"]
    new_maps = [dict(m) if isinstance(m, dict) else {} for m in fatigue_col]
    cid_to_idx = {str(cid): i for i, cid in enumerate(customers_df["customer_id"].astype(str))}
    for _, row in exposure_df.iterrows():
        cid = row.get("customer_id")
        crid = row.get("creative_id")
        if pd.isna(crid) or cid is None:
            continue
        cid_str = str(cid)
        if cid_str not in cid_to_idx:
            continue
        idx = cid_to_idx[cid_str]
        crid_str = str(crid)
        increment = creative_fatigue_mult.get(crid_str, 1.0)
        new_maps[idx][crid_str] = new_maps[idx].get(crid_str, 0) + increment
    for i in range(len(new_maps)):
        m = new_maps[i]
        for k in list(m.keys()):
            m[k] *= decay
            if m[k] < thresh:
                del m[k]
    customers_df["creative_fatigue_map"] = new_maps
    return customers_df


def _update_discount_dependency(
    customers_df: pd.DataFrame, orders_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Price psychology: add discount_pct from today's orders to discount_dependency, then daily decay."""
    if "discount_dependency" not in customers_df.columns:
        return customers_df
    customers_df = customers_df.copy()
    dep = customers_df["discount_dependency"].fillna(0).values.copy()
    if not orders_df.empty and "discount_pct" in orders_df.columns:
        cid_to_idx = {str(cid): i for i, cid in enumerate(customers_df["customer_id"].astype(str))}
        for _, row in orders_df.iterrows():
            cid = row.get("customer_id")
            pct = row.get("discount_pct", 0) or 0
            if pd.isna(pct) or float(pct) <= 0:
                continue
            cid_str = str(cid)
            if cid_str in cid_to_idx:
                dep[cid_to_idx[cid_str]] += float(pct)
    decay = float(config.get("dependency_decay", 0.97))
    dep = dep * decay
    customers_df["discount_dependency"] = dep
    return customers_df


def _update_last_exposure_date(
    customers_df: pd.DataFrame, exposure_df: pd.DataFrame, date_str: str
) -> pd.DataFrame:
    """Set last_exposure_date to date_str for every customer who received exposure today."""
    if exposure_df.empty:
        return customers_df
    customers_df = customers_df.copy()
    if "last_exposure_date" not in customers_df.columns:
        customers_df["last_exposure_date"] = pd.NaT
    exposed = exposure_df["customer_id"].drop_duplicates()
    date_ts = pd.to_datetime(date_str).normalize()
    mask = customers_df["customer_id"].astype(str).isin(exposed.astype(str))
    customers_df.loc[mask, "last_exposure_date"] = date_ts
    return customers_df


def _increment_exposure_count(
    customers_df: pd.DataFrame, exposure_df: pd.DataFrame
) -> pd.DataFrame:
    """Tier 5: accumulate per-customer exposure count so drift strength can scale with history."""
    if exposure_df.empty or "exposure_count" not in customers_df.columns:
        return customers_df
    counts = exposure_df.groupby("customer_id").size().reset_index(name="_day_count")
    customers_df = customers_df.copy()
    left = customers_df[["customer_id", "exposure_count"]].copy()
    left["_cid"] = left["customer_id"].astype(str)
    counts["_cid"] = counts["customer_id"].astype(str)
    merged = left.merge(counts[["_cid", "_day_count"]], on="_cid", how="left")
    merged["_day_count"] = merged["_day_count"].fillna(0).astype("int64")
    customers_df["exposure_count"] = merged["exposure_count"].values + merged["_day_count"].values
    return customers_df


# --- Expressed desire (energy variables only; no labels) ---
def _update_desire_after_exposure(
    customers_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    creatives_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Exposure creates desire: delta = base_exposure_lift + promise_intensity_weighted_sum per exposure."""
    if exposure_df.empty or "expressed_desire_level" not in customers_df.columns:
        return customers_df
    base_lift = float(config.get("base_exposure_lift", 0.02))
    # promise_energy weights: urgency 0.4, premium 0.3, discount 0.2, safety 0.1
    promise_cols = ["promise_urgency_focus", "promise_premium_focus", "promise_discount_focus", "promise_safety_focus"]
    if not creatives_df.empty and all(c in creatives_df.columns for c in promise_cols):
        exp = exposure_df.merge(
            creatives_df[["creative_id"] + promise_cols],
            on="creative_id",
            how="left",
        )
        for c in promise_cols:
            exp[c] = exp[c].fillna(0.5)
        exp["_promise_energy"] = (
            0.4 * exp["promise_urgency_focus"].values
            + 0.3 * exp["promise_premium_focus"].values
            + 0.2 * exp["promise_discount_focus"].values
            + 0.1 * exp["promise_safety_focus"].values
        )
    else:
        exp = exposure_df.copy()
        exp["_promise_energy"] = 0.5
    exp["_delta"] = base_lift + exp["_promise_energy"].values
    per_customer = exp.groupby("customer_id")["_delta"].sum().reset_index()
    per_customer["customer_id"] = per_customer["customer_id"].astype(str)
    customers_df = customers_df.copy()
    cid_str = customers_df["customer_id"].astype(str)
    lookup = per_customer.set_index("customer_id")["_delta"].to_dict()
    add = cid_str.map(lookup).fillna(0).values
    customers_df["expressed_desire_level"] = (
        customers_df["expressed_desire_level"].fillna(0.1).values + add
    )
    customers_df["expressed_desire_level"] = customers_df["expressed_desire_level"].clip(upper=1.0)
    return customers_df


def _apply_click_desire_bonus(
    customers_df: pd.DataFrame, clicks_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """When click occurs: expressed_desire_level += click_desire_bonus."""
    if clicks_df.empty or "expressed_desire_level" not in customers_df.columns:
        return customers_df
    bonus = float(config.get("click_desire_bonus", 0.05))
    counts = clicks_df.groupby("customer_id").size().reset_index(name="_clicks")
    counts["customer_id"] = counts["customer_id"].astype(str)
    customers_df = customers_df.copy()
    cid_str = customers_df["customer_id"].astype(str)
    lookup = counts.set_index("customer_id")["_clicks"].to_dict()
    n_clicks = cid_str.map(lookup).fillna(0).values.astype(int)
    customers_df["expressed_desire_level"] = (
        customers_df["expressed_desire_level"].fillna(0.1).values + n_clicks * bonus
    )
    customers_df["expressed_desire_level"] = customers_df["expressed_desire_level"].clip(upper=1.0)
    return customers_df


def _apply_page_experience_trust_update(
    customers_df: pd.DataFrame,
    clicks_df: pd.DataFrame,
    funnel_events_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Tier 8: pre-purchase trust and memory micro-adjustments from page experience.

    Humans who clicked but bounced (no landing_page_view event) experienced a negative
    page encounter — small trust decay and disappointment nudge.
    Humans who stayed (got LPV event) experienced a coherent page — slight trust reinforcement.
    This happens before any purchase, making trust curves react to page quality.
    """
    if clicks_df.empty:
        return customers_df
    for col in ["trust_score", "disappointment_memory", "recent_negative_velocity"]:
        if col not in customers_df.columns:
            return customers_df

    bounce_trust_decay = float(config.get("page_bounce_trust_decay", 0.008))
    bounce_disappointment_nudge = float(config.get("page_bounce_disappointment_nudge", 0.01))
    bounce_neg_vel_nudge = float(config.get("page_bounce_neg_vel_nudge", 0.005))
    stay_trust_boost = float(config.get("page_stay_trust_boost", 0.003))

    clicker_ids = set(clicks_df["customer_id"].astype(str))
    lpv_ids = set()
    if (
        funnel_events_df is not None
        and not funnel_events_df.empty
        and "event_type" in funnel_events_df.columns
        and "customer_id" in funnel_events_df.columns
    ):
        lpv_mask = funnel_events_df["event_type"] == "landing_page_view"
        lpv_ids = set(funnel_events_df.loc[lpv_mask, "customer_id"].astype(str))

    bounced_ids = clicker_ids - lpv_ids
    stayed_ids = clicker_ids & lpv_ids

    if not bounced_ids and not stayed_ids:
        return customers_df

    customers_df = customers_df.copy()
    cid_str = customers_df["customer_id"].astype(str)

    if bounced_ids:
        bounce_mask = cid_str.isin(bounced_ids).values
        if bounce_mask.any():
            customers_df.loc[bounce_mask, "trust_score"] = np.clip(
                customers_df.loc[bounce_mask, "trust_score"].values - bounce_trust_decay,
                0.0, 1.0,
            )
            customers_df.loc[bounce_mask, "disappointment_memory"] = (
                customers_df.loc[bounce_mask, "disappointment_memory"].values
                + bounce_disappointment_nudge
            )
            customers_df.loc[bounce_mask, "recent_negative_velocity"] = (
                customers_df.loc[bounce_mask, "recent_negative_velocity"].values
                + bounce_neg_vel_nudge
            )

    if stayed_ids:
        stay_mask = cid_str.isin(stayed_ids).values
        if stay_mask.any():
            customers_df.loc[stay_mask, "trust_score"] = np.clip(
                customers_df.loc[stay_mask, "trust_score"].values + stay_trust_boost,
                0.0, 1.0,
            )

    return customers_df


def _apply_desire_daily_decay(customers_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """End of day: expressed_desire_level *= desire_decay."""
    if "expressed_desire_level" not in customers_df.columns:
        return customers_df
    decay = float(config.get("desire_decay", 0.97))
    customers_df = customers_df.copy()
    customers_df["expressed_desire_level"] = (
        customers_df["expressed_desire_level"].fillna(0.1) * decay
    )
    return customers_df


def _apply_disappointment_desire_collapse(
    customers_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """If strong disappointment: expressed_desire_level *= disappointment_desire_collapse."""
    if "expressed_desire_level" not in customers_df.columns or "disappointment_memory" not in customers_df.columns:
        return customers_df
    thresh = float(config.get("disappointment_desire_threshold", 0.35))
    collapse = float(config.get("disappointment_desire_collapse", 0.85))
    customers_df = customers_df.copy()
    strong = customers_df["disappointment_memory"].fillna(0).values > thresh
    if strong.any():
        desire = customers_df["expressed_desire_level"].values.copy()
        desire[strong] *= collapse
        customers_df["expressed_desire_level"] = desire
    return customers_df


def _update_customers_last_order_from_orders(
    customers_df: pd.DataFrame, orders_df: pd.DataFrame, date_str: str
) -> pd.DataFrame:
    """
    Tier 3: set last_order_date and last_attributed_* from today's orders.
    For each customer who ordered, use their last order (by row order) for attribution.
    """
    if orders_df.empty:
        return customers_df
    if "customer_id" not in orders_df.columns:
        return customers_df
    attr_cols = ["last_attributed_ad_id", "last_attributed_adset_id", "last_attributed_campaign_id", "last_attributed_creative_id"]
    # Last order per customer (last row wins)
    last_per_customer = orders_df.drop_duplicates(subset=["customer_id"], keep="last")
    date_ts = pd.Timestamp(date_str).normalize()
    customers_df = customers_df.copy()
    cid_str = customers_df["customer_id"].astype(str)
    last_cid_str = set(last_per_customer["customer_id"].astype(str))
    idx = cid_str.isin(last_cid_str)
    if not idx.any():
        return customers_df
    lookup = last_per_customer.set_index("customer_id")
    for attr in attr_cols:
        if attr not in last_per_customer.columns:
            continue
        if attr not in customers_df.columns:
            continue
        customers_df.loc[idx, attr] = (
            customers_df.loc[idx, "customer_id"].map(lookup[attr].to_dict())
        )
    customers_df.loc[idx, "last_order_date"] = date_ts
    return customers_df


def _enrich_performance_with_hierarchy(
    performance_df: pd.DataFrame, ads_df: pd.DataFrame
) -> pd.DataFrame:
    """Add adset_id, campaign_id, account_id to meta_ad_performance_daily from ads_df. Overwrites if present."""
    hierarchy_cols = ["adset_id", "campaign_id", "account_id"]
    out = performance_df.drop(columns=[c for c in hierarchy_cols if c in performance_df.columns])
    ads_lookup = ads_df[["id"] + hierarchy_cols].rename(columns={"id": "_ad_id_key"})
    out = out.merge(ads_lookup, left_on="ad_id", right_on="_ad_id_key", how="left").drop(
        columns=["_ad_id_key"]
    )
    return out

SUBFOLDERS = (
    "meta_exposures",
    "meta_ad_performance_daily",
    "orders",
    "line_items",
    "transactions",
    "refunds",
    "fulfillments",
    "shopify_checkouts",
)


def _ensure_output_dirs() -> None:
    """Create output/ and subfolders if they do not exist (append-safe)."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in SUBFOLDERS:
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def _set_cart_memory(df: pd.DataFrame, abandoned_carts_df: pd.DataFrame, date_str: str) -> None:
    """Set cart_memory dict on humans whose customer_id appears in abandoned_carts_df."""
    if abandoned_carts_df is None or abandoned_carts_df.empty or "cart_memory" not in df.columns:
        return
    cid_col = "customer_id"
    if cid_col not in abandoned_carts_df.columns:
        return
    abandon_map = {}
    for _, row in abandoned_carts_df.iterrows():
        abandon_map[str(row[cid_col])] = {
            "cart_date": date_str,
            "ad_id": row.get("ad_id"),
            "adset_id": row.get("adset_id"),
            "campaign_id": row.get("campaign_id"),
            "creative_id": row.get("creative_id"),
            "discount_pct": float(row.get("_discount_pct", 0.0) or 0.0),
            "discount_code": row.get("_discount_code"),
            "alignment": float(row.get("_alignment", 0.5) or 0.5),
            "cart_age_days": 0,
        }
    for idx in df.index:
        cid = str(df.at[idx, cid_col])
        if cid in abandon_map and df.at[idx, "cart_memory"] is None:
            df.at[idx, "cart_memory"] = abandon_map[cid]


def _clear_cart_memory(df: pd.DataFrame, purchased_ids: set) -> None:
    """Clear cart_memory for humans who purchased today."""
    if "cart_memory" not in df.columns:
        return
    for idx in df.index:
        cid = str(df.at[idx, "customer_id"])
        if cid in purchased_ids and df.at[idx, "cart_memory"] is not None:
            df.at[idx, "cart_memory"] = None


def _age_and_expire_cart_memory(df: pd.DataFrame, expiry_days: int = 14) -> None:
    """Increment cart_age_days by 1; expire carts older than expiry_days."""
    if "cart_memory" not in df.columns:
        return
    for idx in df.index:
        cart = df.at[idx, "cart_memory"]
        if cart is not None and isinstance(cart, dict):
            cart["cart_age_days"] = cart.get("cart_age_days", 0) + 1
            if cart["cart_age_days"] >= expiry_days:
                df.at[idx, "cart_memory"] = None


def _write_parquet(df, folder: str, date_str: str) -> None:
    path = OUTPUT_ROOT / folder / f"{date_str}.parquet"
    df.to_parquet(path, index=False)


def _build_meta_exposures_for_warehouse(
    exposure_df: pd.DataFrame, customers_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build meta_exposures table for warehouse: all exposures with identity split.
    - Customer exposures: customer_id set, anonymous_id null.
    - Prospect exposures: anonymous_id = prospect_id (stable, deterministic), customer_id null.
    Historical exposures are never backfilled when a prospect converts; they keep anonymous_id.
    """
    if exposure_df.empty:
        out = exposure_df.copy()
        if "customer_id" in out.columns:
            out["anonymous_id"] = pd.NA
        else:
            out["customer_id"] = pd.NA
            out["anonymous_id"] = pd.NA
        return out
    customer_ids_set = set(customers_df["customer_id"].astype(str))
    out = exposure_df.copy()
    audience = out["customer_id"].astype(str)
    is_customer = audience.isin(customer_ids_set)
    out["anonymous_id"] = pd.NA
    out.loc[~is_customer, "anonymous_id"] = out.loc[~is_customer, "customer_id"].astype(object)
    out.loc[~is_customer, "customer_id"] = pd.NA
    return out


def _validate_meta_exposures_identity(exposure_for_write: pd.DataFrame) -> None:
    """Every exposure row must have exactly one of customer_id, anonymous_id (non-null)."""
    if exposure_for_write.empty:
        return
    has_customer = exposure_for_write["customer_id"].notna()
    has_anonymous = exposure_for_write["anonymous_id"].notna()
    exactly_one = (has_customer & ~has_anonymous) | (~has_customer & has_anonymous)
    if not exactly_one.all():
        bad = ~exactly_one
        n = int(bad.sum())
        raise ValueError(
            f"meta_exposures identity violation: {n} row(s) must have exactly one of "
            "customer_id or anonymous_id non-null (not both, not neither)."
        )


def _print_video_watch_distribution(start_str: str, end_str: str) -> None:
    """Load written meta_exposures, filter to video, print watch distribution and thruplay rate."""
    folder = OUTPUT_ROOT / "meta_exposures"
    if not folder.is_dir():
        return
    start_d = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_d = datetime.strptime(end_str, "%Y-%m-%d").date()
    dfs = []
    d = start_d
    while d <= end_d:
        path = folder / f"{d.isoformat()}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
        d += timedelta(days=1)
    if not dfs:
        return
    all_exp = pd.concat(dfs, ignore_index=True)
    if "creative_type" not in all_exp.columns or "watch_time_seconds" not in all_exp.columns:
        return
    from generators.meta import _is_video_type
    is_vid = all_exp["creative_type"].map(lambda ct: _is_video_type(ct) if pd.notna(ct) else False)
    video = all_exp[is_vid & (all_exp["video_duration_seconds"].fillna(0) > 0)].copy() if "video_duration_seconds" in all_exp.columns else all_exp[is_vid].copy()
    if video.empty:
        print("Video watch distribution: no video exposures in written meta_exposures.")
        return
    watch = video["watch_time_seconds"].fillna(0).values
    duration = video["video_duration_seconds"].fillna(0).values if "video_duration_seconds" in video.columns else np.zeros(len(video))
    n = len(video)
    pct_lt_3 = 100.0 * (watch < 3).sum() / n
    pct_3_10 = 100.0 * ((watch >= 3) & (watch <= 10)).sum() / n
    threshold = np.where(duration > 0, np.maximum(15.0, 0.5 * duration), 15.0)
    thruplay = (watch >= threshold).sum()
    pct_gt_half = 100.0 * ((duration > 0) & (watch >= 0.5 * duration)).sum() / n
    thruplay_rate = 100.0 * thruplay / n
    print("Video watch retention (meta_exposures):")
    print(f"  % watch < 3 sec:     {pct_lt_3:.1f}%")
    print(f"  % watch 3–10 sec:   {pct_3_10:.1f}%")
    print(f"  % watch > 50%% duration: {pct_gt_half:.1f}%")
    print(f"  thruplay rate:      {thruplay_rate:.1f}%")


def _enrich_customers_last_order_at(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Set last_order_at = MAX(created_at) from shopify_orders per customer.
    Uses all parquet files under output/orders/. No orders → last_order_at NULL.
    """
    orders_dir = OUTPUT_ROOT / "orders"
    parquet_files = list(orders_dir.glob("*.parquet")) if orders_dir.exists() else []
    if not parquet_files:
        customers_df = customers_df.copy()
        customers_df["last_order_at"] = pd.NA
        print("customers with last_order_at:\n0")
        return customers_df

    order_dfs = [pd.read_parquet(p) for p in parquet_files]
    orders_df = pd.concat(order_dfs, ignore_index=True)
    if orders_df.empty or "customer_id" not in orders_df.columns or "created_at" not in orders_df.columns:
        customers_df = customers_df.copy()
        customers_df["last_order_at"] = pd.NA
        print("customers with last_order_at:\n0")
        return customers_df

    orders_df["created_at"] = pd.to_datetime(orders_df["created_at"])
    max_orders = orders_df.groupby("customer_id")["created_at"].max().reset_index()
    max_orders = max_orders.rename(columns={"created_at": "last_order_at"})

    customers_df = customers_df.drop(columns=["last_order_at"], errors="ignore")
    customers_df = customers_df.merge(max_orders, on="customer_id", how="left")
    n_with = customers_df["last_order_at"].notna().sum()
    print("customers with last_order_at:\n" + str(n_with))
    return customers_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-day behavioral simulation.")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to config YAML (default: config.yaml in script directory)",
    )
    args = parser.parse_args()
    _config_dir = Path(__file__).resolve().parent
    config_path = (
        Path(args.config).resolve()
        if args.config
        else _config_dir / "config.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    sim = config.get("simulation") or {}
    start_str = sim.get("start_date", "2023-01-01")
    end_str = sim.get("end_date", "2026-01-01")  # default 3 years
    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    if start_date > end_date:
        raise ValueError(f"start_date {start_str} must be <= end_date {end_str}")

    num_days = (end_date - start_date).days + 1
    print(
        f"Simulation: {start_str} to {end_str} ({num_days} days) [config: {config_path.resolve()}]"
    )

    _ensure_output_dirs()

    # --- Setup (once): static entities ---
    # First-purchase-creates-customer: prospects (anonymous) see ads and click; conversion creates customer.
    prospects_df = generate_prospects(config)
    customers_df = get_empty_customers_schema(config)
    products_df = generate_products(config)
    variants_df = generate_variants(products_df)
    _assert_shopify_static_hierarchy(products_df, variants_df)
    creatives_df = generate_creatives(config)
    ad_accounts_df = generate_ad_accounts(config)
    campaigns_df = generate_campaigns(creatives_df, ad_accounts_df, config)
    adsets_df = generate_adsets(campaigns_df, config)
    ads_df = generate_ads(adsets_df, creatives_df)
    _require_ads_hierarchy(ads_df)  # fail fast: every ad must have creative_id

    # Propagate Meta hierarchy downward (no generator changes; enrich via merges)
    adsets_df = _enrich_adsets_with_account_id(adsets_df, campaigns_df)
    ads_df = _enrich_ads_with_campaign_and_account(ads_df, adsets_df)
    ads_df = _enrich_ads_with_flight_dates(ads_df, adsets_df)
    ads_df = _enrich_ads_with_creative_name_and_type(ads_df, creatives_df)
    ads_df = _enrich_ads_with_audience_type_and_objective(ads_df, adsets_df, campaigns_df)

    # Upgrade 1: Brand lifecycle state (phase effects, accumulators, transitions)
    brand_state = BrandState(config)

    # Upgrade 2: Population dynamics (prospect pool inflow/outflow/drift)
    pop_manager = PopulationManager(prospects_df, config, start_date)

    # Upgrade 4: Momentum & seasonality tracker
    momentum = MomentumTracker(config)

    # Candidate pool: prospects + customers. Exposure on both; first purchase creates customer.
    candidates_df = pd.concat([prospects_df, customers_df], ignore_index=True)

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")

        # Upgrade 2: Update prospect pool (add new, cull expired)
        prospects_df = pop_manager.daily_update(current, brand_state=brand_state)

        # Upgrade 4: Compute daily seasonality + momentum multipliers
        seasonality_mult = momentum.get_seasonality_multiplier(current)
        ctr_momentum = momentum.get_rate_modifier("ctr")
        cvr_momentum = momentum.get_rate_modifier("cvr")
        # Combine seasonality with CTR momentum for exposure; CVR momentum applied separately
        exposure_mult = seasonality_mult * ctr_momentum
        conversion_mult = seasonality_mult * cvr_momentum

        last_exposure_by_customer = None
        if "last_exposure_date" in candidates_df.columns:
            last_exposure_by_customer = candidates_df.set_index("customer_id")["last_exposure_date"].to_dict()
        exposure_df = simulate_daily_exposure(
            candidates_df, ads_df, creatives_df, date_str, config,
            last_exposure_by_customer=last_exposure_by_customer,
            brand_state=brand_state,
            seasonality_mult=exposure_mult,
        )
        n_exposures = len(exposure_df)

        # Only customers get state updates (fatigue, last_exposure, drift, desire)
        exposure_customers_only = exposure_df[exposure_df["customer_id"].astype(str).isin(customers_df["customer_id"].astype(str))]
        customers_df = _update_fatigue_after_exposure(customers_df, exposure_customers_only, config, creatives_df=creatives_df)
        customers_df = _update_last_exposure_date(customers_df, exposure_customers_only, date_str)
        customers_df = apply_identity_drift(customers_df, exposure_customers_only, creatives_df, config)
        customers_df = _increment_exposure_count(customers_df, exposure_customers_only)
        customers_df = _update_desire_after_exposure(
            customers_df, exposure_customers_only, creatives_df, config
        )
        candidates_df = pd.concat([prospects_df, customers_df], ignore_index=True)

        clicks = exposure_df[exposure_df["clicked"] == 1]
        n_clicks = len(clicks)
        customers_df = _apply_click_desire_bonus(customers_df, clicks, config)

        orders_df, line_items_df, transactions_df, new_customers_df, funnel_events_df, checkouts_df, abandoned_carts_df = simulate_purchases_from_clicks(
            clicks, customers_df, variants_df, products_df, config, creatives_df, ads_df, prospects_df=prospects_df, brand_state=brand_state, seasonality_mult=conversion_mult,
        )
        if not new_customers_df.empty:
            customers_df = pd.concat([customers_df, new_customers_df], ignore_index=True)
            converted_prospect_ids = new_customers_df["customer_id"].str.replace("cust_", "", 1)
            prospects_df = prospects_df[~prospects_df["customer_id"].astype(str).isin(converted_prospect_ids)]
            # Upgrade 2: sync population manager
            pop_manager.prospects_df = prospects_df

        # Tier 8: pre-purchase trust/memory micro-adjustments from page experience
        customers_df = _apply_page_experience_trust_update(
            customers_df, clicks, funnel_events_df, config
        )

        # Tier 6: set cart_memory on humans who abandoned carts today
        if not abandoned_carts_df.empty:
            _set_cart_memory(customers_df, abandoned_carts_df, date_str)
            _set_cart_memory(prospects_df, abandoned_carts_df, date_str)

        # Tier 6: simulate cart returns (humans with active cart_memory re-enter checkout)
        candidates_df = pd.concat([prospects_df, customers_df], ignore_index=True)
        cart_ret_orders, cart_ret_li, cart_ret_tx, cart_ret_new_cust, cart_ret_checkouts, cart_ret_funnel = simulate_cart_returns(
            candidates_df, variants_df, products_df, config, pd.Timestamp(date_str), creatives_df
        )
        if not cart_ret_new_cust.empty:
            customers_df = pd.concat([customers_df, cart_ret_new_cust], ignore_index=True)
            converted_prospect_ids = cart_ret_new_cust["customer_id"].str.replace("cust_", "", 1)
            prospects_df = prospects_df[~prospects_df["customer_id"].astype(str).isin(converted_prospect_ids)]
            # Upgrade 2: sync population manager
            pop_manager.prospects_df = prospects_df
        if not cart_ret_orders.empty:
            orders_df = pd.concat([orders_df, cart_ret_orders], ignore_index=True)
            line_items_df = pd.concat([line_items_df, cart_ret_li], ignore_index=True)
            transactions_df = pd.concat([transactions_df, cart_ret_tx], ignore_index=True)
        if not cart_ret_checkouts.empty:
            checkouts_df = pd.concat([checkouts_df, cart_ret_checkouts], ignore_index=True)
        if not cart_ret_funnel.empty:
            funnel_events_df = pd.concat([funnel_events_df, cart_ret_funnel], ignore_index=True)

        # Tier 6: clear cart_memory for humans who purchased today (fresh funnel or cart return)
        if not orders_df.empty:
            purchased_ids = set(orders_df["customer_id"].astype(str))
            _clear_cart_memory(customers_df, purchased_ids)
            _clear_cart_memory(prospects_df, purchased_ids)

        # Tier 6: age active carts and expire stale ones
        cart_expiry_days = config.get("cart_expiry_days", 14)
        _age_and_expire_cart_memory(customers_df, cart_expiry_days)
        _age_and_expire_cart_memory(prospects_df, cart_expiry_days)

        candidates_df = pd.concat([prospects_df, customers_df], ignore_index=True)

        # Tier 3: repeat purchases (no ad touch); concat with ad-driven orders
        repeat_orders_df, repeat_line_items_df, repeat_transactions_df = simulate_repeat_purchases_for_day(
            customers_df, variants_df, products_df, pd.Timestamp(date_str), config, brand_state=brand_state,
        )
        if not repeat_orders_df.empty:
            orders_df = pd.concat([orders_df, repeat_orders_df], ignore_index=True)
            line_items_df = pd.concat([line_items_df, repeat_line_items_df], ignore_index=True)
            transactions_df = pd.concat([transactions_df, repeat_transactions_df], ignore_index=True)

        n_orders = len(orders_df)
        _assert_shopify_orders_hierarchy(
            orders_df, line_items_df, transactions_df,
            customers_df, products_df, variants_df,
        )

        # Meta performance = pure aggregation of today's events (same causal world as Shopify).
        # Invariant: reported clicks == count(clicks in exposure_df); reported purchases == count(orders by last_attributed_ad_id).
        performance_df = build_ad_performance_daily(
            ads_df, adsets_df, campaigns_df, date_str, config,
            exposure_df=exposure_df,
            orders_df=orders_df,
            funnel_events_df=funnel_events_df,
        )
        performance_df = _enrich_performance_with_hierarchy(performance_df, ads_df)
        _write_parquet(performance_df, "meta_ad_performance_daily", date_str)

        # Update last_order_date and last_attributed_* for customers who ordered today
        customers_df = _update_customers_last_order_from_orders(customers_df, orders_df, date_str)

        # Attribution validation: non-null last_attributed_creative_id must exist in creatives
        if not orders_df.empty and "last_attributed_creative_id" in orders_df.columns:
            non_null = orders_df["last_attributed_creative_id"].dropna()
            if len(non_null) > 0:
                valid_ids = set(creatives_df["creative_id"].astype(str))
                bad = non_null[~non_null.astype(str).isin(valid_ids)]
                if len(bad) > 0:
                    raise ValueError(
                        f"Attribution integrity: {len(bad)} order(s) have last_attributed_creative_id "
                        f"not in creatives: {bad.tolist()[:5]}..."
                    )

        refunds_df = simulate_refunds(
            orders_df,
            line_items_df,
            customers_df,
            products_df,
            variants_df,
            config,
            brand_state=brand_state,
        )
        n_refunds = len(refunds_df)
        _assert_shopify_refunds_hierarchy(refunds_df, orders_df)

        fulfillments_df = simulate_fulfillments(
            orders_df,
            line_items_df,
            products_df,
            variants_df,
            config,
        )
        n_fulfillments = len(fulfillments_df)
        _assert_shopify_fulfillments_hierarchy(fulfillments_df, orders_df)

        # All exposures to warehouse (Meta-real): customers with customer_id, prospects with anonymous_id; no backfill
        exposure_for_write = _build_meta_exposures_for_warehouse(exposure_df, customers_df)
        _validate_meta_exposures_identity(exposure_for_write)
        _write_parquet(exposure_for_write, "meta_exposures", date_str)
        _write_parquet(orders_df, "orders", date_str)
        _write_parquet(line_items_df, "line_items", date_str)
        _write_parquet(transactions_df, "transactions", date_str)
        _write_parquet(refunds_df, "refunds", date_str)
        _write_parquet(fulfillments_df, "fulfillments", date_str)
        _write_parquet(checkouts_df, "shopify_checkouts", date_str)

        # Tier 2: update customer psychological state (expectation vs experience → trust/memory)
        customers_df = update_customers_from_day_orders(
            customers_df,
            orders_df,
            line_items_df,
            products_df,
            fulfillments_df,
            refunds_df,
            creatives_df,
            config,
        )
        # Optional: strong disappointment → desire collapse
        customers_df = _apply_disappointment_desire_collapse(customers_df, config)
        # Price psychology: discount dependency from today's orders + daily decay
        customers_df = _update_discount_dependency(customers_df, orders_df, config)
        # Expressed desire: daily decay
        customers_df = _apply_desire_daily_decay(customers_df, config)

        # Upgrade 6: shipping experience tracking from fulfillments
        customers_df = update_shipping_experience(customers_df, fulfillments_df, orders_df, date_str)

        # Upgrade 6: brand memory decay for inactive customers
        customers_df = apply_brand_memory_decay(customers_df, config)

        # Upgrade 10: Apply trust floor from cumulative refund history
        if "trust_score" in customers_df.columns and brand_state._enabled:
            trust_floor = brand_state.get_trust_floor()
            customers_df["trust_score"] = customers_df["trust_score"].clip(lower=trust_floor)

        # Upgrade 6: increment days_since_last_interaction for all; reset for those with interaction today
        if "days_since_last_interaction" in customers_df.columns:
            customers_df = customers_df.copy()
            customers_df["days_since_last_interaction"] = customers_df["days_since_last_interaction"].fillna(0).values + 1
            # Reset for customers who had exposure or order today
            interacted_ids = set()
            if not exposure_customers_only.empty:
                interacted_ids.update(exposure_customers_only["customer_id"].astype(str))
            if not orders_df.empty:
                interacted_ids.update(orders_df["customer_id"].astype(str))
            if interacted_ids:
                mask = customers_df["customer_id"].astype(str).isin(interacted_ids)
                customers_df.loc[mask, "days_since_last_interaction"] = 0

        # Upgrade 6: update discount_only_buyer flag from today's orders
        if "discount_only_buyer" in customers_df.columns and not orders_df.empty:
            for _, order in orders_df.iterrows():
                cid = str(order.get("customer_id", ""))
                cid_mask = customers_df["customer_id"].astype(str) == cid
                if not cid_mask.any():
                    continue
                idx = cid_mask.idxmax()
                disc_pct = float(order.get("discount_pct", 0) or 0)
                # Track: count total orders and discounted orders via satisfaction_memory as proxy
                # Simple heuristic: if >80% of this customer's total_discounts > 0, flag as discount_only
                cust_orders = orders_df[orders_df["customer_id"].astype(str) == cid]
                total_count = len(cust_orders)
                disc_count = int((cust_orders.get("total_discounts", pd.Series(dtype=float)).fillna(0) > 0).sum())
                if total_count >= 2 and disc_count / total_count > 0.8:
                    customers_df.at[idx, "discount_only_buyer"] = True

        # Upgrade 1: Update brand lifecycle state from today's metrics
        n_repeat_orders = len(repeat_orders_df) if not repeat_orders_df.empty else 0
        n_new_customers = len(new_customers_df) if not new_customers_df.empty else 0
        day_revenue = float(orders_df["total_price"].sum()) if not orders_df.empty and "total_price" in orders_df.columns else 0.0
        day_spend = float(performance_df["spend"].sum()) if not performance_df.empty and "spend" in performance_df.columns else 0.0
        discount_orders_count = int((orders_df["total_discounts"].fillna(0) > 0).sum()) if not orders_df.empty and "total_discounts" in orders_df.columns else 0
        brand_state.update_daily(
            {
                "revenue": day_revenue,
                "orders": n_orders,
                "refunds": n_refunds,
                "new_customers": n_new_customers,
                "repeat_orders": n_repeat_orders,
                "spend": day_spend,
                "discount_orders": discount_orders_count,
            },
            customers_df=customers_df,
        )

        # Upgrade 4: Update momentum EMAs with today's actuals
        momentum.update({
            "revenue": day_revenue,
            "ctr": n_clicks / max(n_exposures, 1),
            "cvr": n_orders / max(n_clicks, 1),
        })

        # Upgrade 9: Update creative lifecycle state (impressions, stage, performance_multiplier)
        old_exhausted = set(creatives_df.loc[creatives_df["lifecycle_stage"] == "exhausted", "creative_id"].astype(str)) if "lifecycle_stage" in creatives_df.columns else set()
        creatives_df = update_creative_lifecycle(creatives_df, exposure_df, config)
        # Upgrade 10: Count newly exhausted creatives → brand-level creative churn
        if "lifecycle_stage" in creatives_df.columns:
            new_exhausted = set(creatives_df.loc[creatives_df["lifecycle_stage"] == "exhausted", "creative_id"].astype(str))
            newly_churned = len(new_exhausted - old_exhausted)
            if newly_churned > 0:
                brand_state.cumulative_creative_churn += newly_churned

        print(
            f"{date_str}\t phase={brand_state.lifecycle_phase.value}\t exposures={n_exposures}\t clicks={n_clicks}\t "
            f"orders={n_orders}\t refunds={n_refunds}\t fulfillments={n_fulfillments}"
        )

        current += timedelta(days=1)

    # First-touch is set at customer creation (purchase-causing exposure); no end-of-run attribution.

    # First-touch hierarchy integrity: first_touch_* must match meta_ads lookup for first_touch_ad_id
    if "first_touch_ad_id" in customers_df.columns and customers_df["first_touch_ad_id"].notna().any():
        ads_lookup = ads_df[["id", "adset_id", "campaign_id", "account_id"]].copy()
        ads_lookup["id"] = ads_lookup["id"].astype(str)
        ft = customers_df.loc[customers_df["first_touch_ad_id"].notna(), ["first_touch_ad_id", "first_touch_adset_id", "first_touch_campaign_id"]].copy()
        ft["_ad_id"] = ft["first_touch_ad_id"].astype(str)
        ft = ft.merge(ads_lookup, left_on="_ad_id", right_on="id", how="left")
        if ft["id"].isna().any():
            missing = ft.loc[ft["id"].isna(), "_ad_id"].unique().tolist()[:5]
            raise ValueError(f"First-touch integrity: some first_touch_ad_id not found in meta_ads: {missing}...")
        # When we have hierarchy fields, require they match meta_ads
        if "first_touch_adset_id" in customers_df.columns:
            bad = ft.loc[ft["first_touch_adset_id"].notna() & (ft["first_touch_adset_id"].astype(str) != ft["adset_id"].astype(str))]
            if len(bad) > 0:
                raise ValueError(f"First-touch integrity: first_touch_adset_id mismatch for {len(bad)} row(s).")
        if "first_touch_campaign_id" in customers_df.columns:
            bad = ft.loc[ft["first_touch_campaign_id"].notna() & (ft["first_touch_campaign_id"].astype(str) != ft["campaign_id"].astype(str))]
            if len(bad) > 0:
                raise ValueError(f"First-touch integrity: first_touch_campaign_id mismatch for {len(bad)} row(s).")

    # Attribution validation: every first_touch_creative_id on customers must exist in creatives
    if "first_touch_creative_id" in customers_df.columns:
        non_null = customers_df["first_touch_creative_id"].dropna()
        if len(non_null) > 0:
            valid_ids = set(creatives_df["creative_id"].astype(str))
            bad = non_null[~non_null.astype(str).isin(valid_ids)]
            if len(bad) > 0:
                raise ValueError(
                    f"First-touch integrity: {len(bad)} customer(s) have first_touch_creative_id "
                    f"not in creatives: {bad.tolist()[:5]}..."
                )

    # Temporal sanity: first-touch cannot occur after a customer's last_order_date (if any)
    if "first_touch_date" in customers_df.columns and "last_order_date" in customers_df.columns:
        ft = pd.to_datetime(customers_df["first_touch_date"], errors="coerce")
        lo = pd.to_datetime(customers_df["last_order_date"], errors="coerce")
        bad = ft.notna() & lo.notna() & (ft > lo)
        if bad.any():
            n = int(bad.sum())
            sample = customers_df.loc[bad, ["customer_id", "first_touch_date", "last_order_date"]].head(5).to_dict("records")
            raise ValueError(f"First-touch temporal integrity: {n} customer(s) have first_touch_date > last_order_date. Sample: {sample}")

    # Safety: first_touch_date must never be after signup_date (chronological first-touch logic guarantees this)
    if not customers_df.empty and "signup_date" in customers_df.columns and "first_touch_date" in customers_df.columns:
        signup = pd.to_datetime(customers_df["signup_date"], errors="coerce").dt.normalize()
        first_touch = pd.to_datetime(customers_df["first_touch_date"], errors="coerce")
        bad_ft = first_touch.notna() & (first_touch.dt.normalize() > signup)
        if bad_ft.any():
            n = int(bad_ft.sum())
            raise ValueError(
                f"First-touch chronology violation: {n} customer(s) have first_touch_date > signup_date. "
                "This should never happen if acquisition logic is correct."
            )

    # Video watch retention distribution (from written meta_exposures)
    _print_video_watch_distribution(start_str, end_str)

    # Enrich customers.last_order_at from all orders (after daily loop, before BQ write)
    customers_df = _enrich_customers_last_order_at(customers_df)

    project_id = config.get("project_id")
    dataset_meta = config.get("dataset_meta")
    dataset_shopify = config.get("dataset_shopify")
    if project_id and dataset_meta and dataset_shopify:
        bq_client = bigquery.Client(project=project_id)
        # Visibility layer: expose behavioral memory; drop non-serializable columns only
        customers_for_bq = customers_df.drop(
            columns=[
                "creative_fatigue_map", "cart_memory",
                "shipping_bad_count", "shipping_good_count",
                "discount_only_buyer", "days_since_last_interaction",
                "spending_propensity", "_pool_entry_date",
            ],
            errors="ignore",
        )
        # Safety defaults for visibility columns (old configs / partial flows)
        for col, default in [
            ("recent_negative_velocity", 0.0),
            ("discount_dependency", 0.0),
            ("exposure_count", 0),
            ("expressed_desire_level", 0.1),
            ("desire_decay_memory", 0.0),
        ]:
            if col not in customers_for_bq.columns:
                customers_for_bq[col] = default
            else:
                customers_for_bq[col] = customers_for_bq[col].fillna(default)
        if "exposure_count" in customers_for_bq.columns:
            customers_for_bq["exposure_count"] = customers_for_bq["exposure_count"].astype("int64")
        # Upgrade 9: drop internal creative lifecycle columns before BQ write
        creatives_for_bq = creatives_df.drop(
            columns=[
                "total_impressions", "days_active",
                "lifecycle_stage", "performance_multiplier",
                "innate_quality",
            ],
            errors="ignore",
        )
        load_static_entities(
            bq_client,
            project_id,
            dataset_meta,
            dataset_shopify,
            customers_for_bq,
            products_df,
            variants_df,
            creatives_for_bq,
            ad_accounts_df,
            campaigns_df,
            adsets_df,
            ads_df,
        )


if __name__ == "__main__":
    main()
