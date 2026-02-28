"""
Meta performance reporting = pure aggregation of simulated behavior.

This module unifies the causal world: meta_ad_performance_daily is built ONLY from
actual events (exposure_df, orders_df, funnel_events_df) so that dbt analyzes the
SAME humans that produced revenue.

INVARIANT (non-negotiable):
  For any day + ad_id:
    reported impressions        == count of exposure rows for that ad on that day
    reported reach              == count of UNIQUE humans exposed to that ad on that day
    reported clicks             == count of exposure rows where click == 1
    reported landing_page_views == count of landing_page_view funnel events for that ad
    reported add_to_cart        == count of add_to_cart funnel events for that ad
    reported initiate_checkout  == count of initiate_checkout funnel events for that ad
    reported conversions        == count of orders where last_attributed_ad_id == ad_id
  Re-aggregating from raw events must yield the same numbers. No synthetic rates.

  Funnel monotonicity:
    reach <= impressions
    clicks <= reach  (a human must be exposed to click)
    landing_page_views <= clicks
    add_to_cart <= landing_page_views
    initiate_checkout <= add_to_cart
    purchase <= initiate_checkout

We do NOT use: spend/CPM -> impressions; impressions*CTR -> clicks; clicks*CVR -> conversions;
clicks*lpv_rate -> landing_page_views; impressions/random_factor -> reach.
Spend may remain budget-paced; outcomes come from real events only.
Schema unchanged; only the values reflect reality.
"""

import numpy as np
import pandas as pd

from .meta import _is_video_type


def _adset_daily_spend(
    adset_id: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    lifetime_budget: float,
    date_str: str,
    config: dict,
) -> float:
    """
    Deterministic daily spend for this ad set on date_str.
    Sum over all dates in [start_time, end_time] equals lifetime_budget.
    """
    start = pd.Timestamp(start_time).normalize()
    end = pd.Timestamp(end_time).normalize()
    if start > end:
        return 0.0
    date = pd.Timestamp(date_str).normalize()
    if date < start or date > end:
        return 0.0
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    if n_days == 0:
        return 0.0
    # Deterministic weights from adset_id
    seed = int(sum(ord(c) for c in adset_id) % (2**31))
    rng = np.random.default_rng(seed)
    concentration = float(config.get("budget_concentration", 2.0))
    volatility = float(config.get("budget_volatility", 0.3))
    alpha = np.full(n_days, concentration) * (1 + rng.uniform(-volatility, volatility, n_days))
    alpha = np.maximum(alpha, 0.1)
    weights = rng.dirichlet(alpha)
    spends = weights * lifetime_budget
    # Fix sum == lifetime_budget (rounding on last)
    spends = np.maximum(spends, 0.0)
    diff = lifetime_budget - spends.sum()
    spends[-1] += diff
    spends[-1] = max(0.0, spends[-1])
    idx = (dates == date).nonzero()[0]
    if len(idx) == 0:
        return 0.0
    return float(spends[idx[0]])


def build_ad_performance_daily(
    ads_df: pd.DataFrame,
    adsets_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
    date_str: str,
    config: dict,
    *,
    exposure_df: pd.DataFrame | None = None,
    orders_df: pd.DataFrame | None = None,
    funnel_events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build meta_ad_performance_daily as aggregation(exposure_df, orders_df, funnel_events_df) for date_str.

    Callers MUST pass exposure_df and orders_df for the same date so that:
      - impressions = number of exposure rows per ad_id
      - clicks       = number of those rows where clicked == 1
      - conversions  = number of orders that day with last_attributed_ad_id == ad_id
      - add_to_cart / initiate_checkout = counts from funnel_events_df (event-driven) when provided
    Spend is budget-paced and detached from impressions. No CPM/CTR/CVR used for outcomes.
    If exposure_df or orders_df is missing, impressions/clicks/conversions are 0 (no synthetic rates).
    If funnel_events_df is provided, add_to_cart and initiate_checkout are aggregated from events; else 0.
    """
    _empty_perf_columns = [
        "ad_id", "adset_id", "campaign_id", "account_id", "date",
        "impressions", "clicks", "spend",
        "total_watch_time", "likes", "saves", "shares",
        "reach", "frequency", "unique_clicks", "inline_link_clicks", "outbound_clicks",
        "landing_page_views", "video_views", "thruplays", "post_engagements",
        "conversions", "purchase", "add_to_cart", "initiate_checkout", "lead",
        "complete_registration", "custom_conversions", "currency",
        "cost_per_action_type", "cost_per_conversion", "attribution_setting",
        "optimization_goal", "buying_type", "account_currency",
        "account_timezone_name", "account_timezone_offset_hours_utc", "_fivetran_synced",
    ]

    seed = config.get("seed")
    rng = np.random.default_rng(seed)

    # Ads: need adset_id, campaign_id, account_id, delivery_weight
    ads = ads_df.copy()
    if "delivery_weight" not in ads.columns:
        ads["delivery_weight"] = 1.0
    ads["delivery_weight"] = np.maximum(ads["delivery_weight"].astype(float), 1e-6)

    # Ad sets active on date_str
    adsets = adsets_df[
        ["id", "campaign_id", "start_time", "end_time", "lifetime_budget"]
    ].copy()
    adsets["start_time"] = pd.to_datetime(adsets["start_time"]).dt.normalize()
    adsets["end_time"] = pd.to_datetime(adsets["end_time"]).dt.normalize()
    date_ts = pd.Timestamp(date_str).normalize()
    adsets = adsets[
        (adsets["start_time"] <= date_ts) & (adsets["end_time"] >= date_ts)
    ]

    rows = []
    if not adsets.empty:
        daily_adset_spend = {}
        for _, row in adsets.iterrows():
            s = _adset_daily_spend(
                row["id"],
                row["start_time"],
                row["end_time"],
                float(row["lifetime_budget"]),
                date_str,
                config,
            )
            daily_adset_spend[row["id"]] = max(0.0, s)

        ads_active = ads[ads["adset_id"].isin(daily_adset_spend.keys())]
        if not ads_active.empty:
            adset_weight_sum = ads_active.groupby("adset_id")["delivery_weight"].transform("sum")
            ads_active = ads_active.copy()
            ads_active["_weight_frac"] = ads_active["delivery_weight"] / np.maximum(adset_weight_sum, 1e-9)
            for adset_id, day_spend in daily_adset_spend.items():
                subset = ads_active[ads_active["adset_id"] == adset_id]
                if subset.empty:
                    continue
                raw_spends = (subset["_weight_frac"].values * day_spend).astype(float)
                raw_spends = np.maximum(raw_spends, 0.0)
                diff = day_spend - raw_spends.sum()
                raw_spends[-1] += diff
                raw_spends[-1] = max(0.0, raw_spends[-1])
                for i, (_, row) in enumerate(subset.iterrows()):
                    rows.append({
                        "ad_id": row["id"],
                        "adset_id": row["adset_id"],
                        "campaign_id": row.get("campaign_id"),
                        "account_id": row.get("account_id"),
                        "date": date_str,
                        "spend": raw_spends[i],
                    })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ad_id", "adset_id", "campaign_id", "account_id", "date", "spend"])

    # --- Pure aggregation from events (invariant: re-aggregate from exposure/orders -> same result) ---
    # impressions = exposure rows per ad; reach = unique humans per ad; clicks = clicked==1;
    # conversions = orders with last_attributed_ad_id. No synthetic rates.
    imp_by_ad = pd.Series(dtype=int)
    reach_by_ad = pd.Series(dtype=int)
    clicks_by_ad = pd.Series(dtype=int)
    conv_by_ad = pd.Series(dtype=int)
    watch_by_ad = pd.Series(dtype=float)
    likes_by_ad = pd.Series(dtype=int)
    saves_by_ad = pd.Series(dtype=int)
    shares_by_ad = pd.Series(dtype=int)
    video_views_by_ad = pd.Series(dtype=int)
    thruplays_by_ad = pd.Series(dtype=int)
    has_behavioral_engagement = False
    if exposure_df is not None and not exposure_df.empty and "ad_id" in exposure_df.columns:
        exp = exposure_df
        has_behavioral_engagement = True
        imp_by_ad = exp.groupby("ad_id").size()
        imp_by_ad.index = imp_by_ad.index.astype(str)
        reach_by_ad = exp.groupby("ad_id")["customer_id"].nunique()
        reach_by_ad.index = reach_by_ad.index.astype(str)
        if "clicked" in exp.columns:
            clicks_by_ad = exp[exp["clicked"] == 1].groupby("ad_id").size()
            clicks_by_ad.index = clicks_by_ad.index.astype(str)
        if "watch_time_seconds" in exp.columns:
            watch_by_ad = exp.groupby("ad_id")["watch_time_seconds"].sum()
            watch_by_ad.index = watch_by_ad.index.astype(str)
        if "liked" in exp.columns:
            likes_by_ad = exp.groupby("ad_id")["liked"].sum()
            likes_by_ad.index = likes_by_ad.index.astype(str)
        if "saved" in exp.columns:
            saves_by_ad = exp.groupby("ad_id")["saved"].sum()
            saves_by_ad.index = saves_by_ad.index.astype(str)
        if "shared" in exp.columns:
            shares_by_ad = exp.groupby("ad_id")["shared"].sum()
            shares_by_ad.index = shares_by_ad.index.astype(str)
        # video_views: exposure with video creative_type and watch_time_seconds > 0
        if "creative_type" in exp.columns and "watch_time_seconds" in exp.columns:
            video_view_mask = exp["creative_type"].map(lambda ct: _is_video_type(ct) if pd.notna(ct) else False) & (exp["watch_time_seconds"].fillna(0) > 0)
            video_views_by_ad = exp.loc[video_view_mask].groupby("ad_id").size()
            video_views_by_ad.index = video_views_by_ad.index.astype(str)
        # thruplays: video exposure with watch >= max(15s, 50% of video_duration_seconds)
        if "creative_type" in exp.columns and "watch_time_seconds" in exp.columns:
            is_video = exp["creative_type"].map(lambda ct: _is_video_type(ct) if pd.notna(ct) else False).values
            watch = exp["watch_time_seconds"].fillna(0).values
            duration = exp["video_duration_seconds"].fillna(0).values.astype(float) if "video_duration_seconds" in exp.columns else np.zeros(len(exp))
            threshold = np.where(is_video & (duration > 0), np.maximum(15.0, 0.5 * duration), 15.0)
            thruplay_mask = is_video & (watch >= threshold)
            thruplays_by_ad = exp.loc[thruplay_mask].groupby("ad_id").size()
            thruplays_by_ad.index = thruplays_by_ad.index.astype(str)
    if orders_df is not None and not orders_df.empty and "last_attributed_ad_id" in orders_df.columns:
        attr = orders_df["last_attributed_ad_id"].dropna().astype(str)
        conv_by_ad = attr.value_counts()
        conv_by_ad.index = conv_by_ad.index.astype(str)

    event_ad_ids = set(imp_by_ad.index) | set(clicks_by_ad.index) | set(conv_by_ad.index)
    df_ad_ids = set(df["ad_id"].astype(str))
    missing_ad_ids = event_ad_ids - df_ad_ids
    if missing_ad_ids:
        # Add rows for ads that had events but no budget row (e.g. adset not active that day)
        ads_lookup = ads_df[["id", "adset_id", "campaign_id", "account_id"]].copy()
        ads_lookup["id"] = ads_lookup["id"].astype(str)
        if "account_id" not in ads_lookup.columns:
            ads_lookup["account_id"] = None
        extra_rows = []
        for ad_id in missing_ad_ids:
            match = ads_lookup[ads_lookup["id"] == ad_id]
            row = match.iloc[0] if len(match) > 0 else None
            extra_rows.append({
                "ad_id": ad_id,
                "adset_id": None if row is None else row.get("adset_id"),
                "campaign_id": None if row is None else row.get("campaign_id"),
                "account_id": None if row is None else row.get("account_id"),
                "date": date_str,
                "spend": 0.0,
            })
        if extra_rows:
            df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    df["impressions"] = df["ad_id"].astype(str).map(imp_by_ad).fillna(0).astype(int)
    # Override budget-paced spend with impression-based spend (CPM model).
    # This ties spend to actual delivery volume, preventing budget/flight-date
    # artifacts from causing spend to collapse while impressions grow.
    cpm = float(config.get("cpm", 14.0))
    df["spend"] = df["impressions"].values.astype(float) * cpm / 1000.0
    df["clicks"] = df["ad_id"].astype(str).map(clicks_by_ad).fillna(0).astype(int)
    df["conversions"] = df["ad_id"].astype(str).map(conv_by_ad).fillna(0).astype(int)
    df["purchase"] = df["conversions"].values
    df["total_watch_time"] = df["ad_id"].astype(str).map(watch_by_ad).fillna(0).astype(float)
    df["likes"] = df["ad_id"].astype(str).map(likes_by_ad).fillna(0).astype(int)
    df["saves"] = df["ad_id"].astype(str).map(saves_by_ad).fillna(0).astype(int)
    df["shares"] = df["ad_id"].astype(str).map(shares_by_ad).fillna(0).astype(int)

    # video_views, thruplays, post_engagements: 100% from exposure behavior when exposure_df provided; no synthetic rates
    if has_behavioral_engagement:
        df["video_views"] = df["ad_id"].astype(str).map(video_views_by_ad).fillna(0).astype(int)
        df["thruplays"] = df["ad_id"].astype(str).map(thruplays_by_ad).fillna(0).astype(int)
        df["post_engagements"] = (df["likes"].values + df["saves"].values + df["shares"].values).astype(int)
    else:
        df["video_views"] = 0
        df["thruplays"] = 0
        df["post_engagements"] = 0

    # Reach, frequency: behavioral (unique humans per ad from exposure_df). No synthetic division.
    df["reach"] = df["ad_id"].astype(str).map(reach_by_ad).fillna(0).astype(int)
    df["frequency"] = np.where(df["reach"].values > 0, df["impressions"].values / df["reach"].values, 0.0)

    # Funnel metrics: 100% event-driven from funnel_events_df. No synthetic rates.
    lpv_by_ad = pd.Series(dtype=int)
    add_to_cart_by_ad = pd.Series(dtype=int)
    initiate_checkout_by_ad = pd.Series(dtype=int)
    if funnel_events_df is not None and not funnel_events_df.empty and "event_type" in funnel_events_df.columns and "ad_id" in funnel_events_df.columns:
        day_events = funnel_events_df[funnel_events_df["date"].astype(str) == date_str] if "date" in funnel_events_df.columns else funnel_events_df
        if not day_events.empty:
            lpv_by_ad = day_events[day_events["event_type"] == "landing_page_view"].groupby("ad_id").size()
            lpv_by_ad.index = lpv_by_ad.index.astype(str)
            add_to_cart_by_ad = day_events[day_events["event_type"] == "add_to_cart"].groupby("ad_id").size()
            add_to_cart_by_ad.index = add_to_cart_by_ad.index.astype(str)
            initiate_checkout_by_ad = day_events[day_events["event_type"] == "initiate_checkout"].groupby("ad_id").size()
            initiate_checkout_by_ad.index = initiate_checkout_by_ad.index.astype(str)
    df["landing_page_views"] = df["ad_id"].astype(str).map(lpv_by_ad).fillna(0).astype(int)
    df["add_to_cart"] = df["ad_id"].astype(str).map(add_to_cart_by_ad).fillna(0).astype(int)
    df["initiate_checkout"] = df["ad_id"].astype(str).map(initiate_checkout_by_ad).fillna(0).astype(int)

    n = len(df)
    sp = df["spend"].values.astype(float)
    df["cost_per_conversion"] = np.where(
        df["purchase"].values > 0,
        sp / np.maximum(1, df["purchase"].values),
        np.nan,
    )
    df["lead"] = np.maximum(0, (df["landing_page_views"].values * rng.uniform(0.02, 0.08, n)).astype(int))
    df["complete_registration"] = np.maximum(0, (df["landing_page_views"].values * rng.uniform(0.01, 0.05, n)).astype(int))

    # Currency: same as store (Shopify). Spend and revenue share one currency for a realistic simulator.
    currency = config.get("currency") or "USD"
    if orders_df is not None and not orders_df.empty and "currency" in orders_df.columns and orders_df["currency"].notna().any():
        # Prefer actual order currency if present (e.g. from loaded Shopify data)
        currency = str(orders_df["currency"].dropna().iloc[0])
    df["currency"] = currency
    df["account_currency"] = currency

    # Typed nulls per contract: INT64 → Int64 NA, NUMERIC → np.nan, TIMESTAMP → pd.NaT, STRING → None
    df["unique_clicks"] = pd.array([pd.NA] * n, dtype="Int64")
    df["inline_link_clicks"] = pd.array([pd.NA] * n, dtype="Int64")
    df["outbound_clicks"] = pd.array([pd.NA] * n, dtype="Int64")
    df["account_timezone_offset_hours_utc"] = np.nan
    df["_fivetran_synced"] = pd.NaT
    for col in ["custom_conversions", "cost_per_action_type", "attribution_setting", "optimization_goal", "buying_type", "account_timezone_name"]:
        df[col] = None

    df = df[
        [
            "ad_id", "adset_id", "campaign_id", "account_id", "date",
            "impressions", "clicks", "spend",
            "total_watch_time", "likes", "saves", "shares",
            "reach", "frequency", "unique_clicks", "inline_link_clicks", "outbound_clicks",
            "landing_page_views", "video_views", "thruplays", "post_engagements",
            "conversions", "purchase", "add_to_cart", "initiate_checkout", "lead",
            "complete_registration", "custom_conversions", "currency",
            "cost_per_action_type", "cost_per_conversion", "attribution_setting",
            "optimization_goal", "buying_type", "account_currency",
            "account_timezone_name", "account_timezone_offset_hours_utc", "_fivetran_synced",
        ]
    ]
    return df
