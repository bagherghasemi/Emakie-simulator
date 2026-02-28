# Benchmark Research Notes

**Created:** 2026-02-26
**Purpose:** Document sources, findings, confidence levels, caveats, and gaps for the simulator structural diagnostics benchmark calibration.
**Target:** DTC Shopify brands selling physical products via Meta ads.

---

## 1. Revenue & Time Series Structure

### 1.1 Revenue Autocorrelation (lag-1)

**Confidence: Medium**

No direct published benchmark for daily ecommerce revenue autocorrelation was found. The calibrated range (0.35-0.80) is derived from first-principles reasoning:

- Deloitte's Retail Volatility Index research shows ecommerce has *lower* daily volatility than physical retail, implying structured (autocorrelated) revenue patterns.
- Meta ad campaigns use daily budget pacing, creating multi-day spend persistence that directly translates to revenue persistence.
- Weekly periodicity (strong lag-7 autocorrelation) creates non-zero lag-1 autocorrelation as well.
- A T-shirt company example showed weekly revenue CV of 25-37.5%, consistent with moderate autocorrelation.

**Gap:** Would benefit from access to actual Shopify merchant daily revenue time series to compute empirical autocorrelation distributions.

### 1.2 Seasonality

**Confidence: High**

Well-documented from multiple sources:

- **Syncio** analysis of $21B in real ecommerce sales: November +29% vs average, February -17% vs average. Best month ~2x worst month.
- **Shopify** Q4 2024 GMV: $94.46B, representing ~26% above average quarter. Q1 consistently weakest.
- **BFCM 2024:** $11.5B GMV in a single weekend (+24% YoY).
- **Weekly patterns:** Oribi data shows Tuesday highest conversion (2.5%), weekend lowest (~2.1%). Amazon confirms Monday-Tuesday peak.

**Caveat:** Seasonality magnitude varies significantly by product category. Gift-heavy products (jewelry, fashion accessories) see much larger Q4 spikes than staple consumables (supplements, personal care).

### 1.3 Trend

**Confidence: Medium**

The simulator explicitly implements a conversion ramp (year 0 to year 2+), so trend is partially mechanistic. External data:

- DTC market growing at ~15% CAGR (2024-2033 projections).
- Shopify merchant GMV grew 26% YoY in 2024.
- Individual DTC brands in growth phase typically see 20-50% annual revenue growth.

**Caveat:** The normalized slope range depends heavily on the specific simulation config (ramp parameters, base conversion rate).

### 1.4 Non-Stationarity

**Confidence: High**

This is a mathematical property, not an empirical benchmark. Any revenue series with growth trend and seasonality will be non-stationary by the ADF test. After differencing, it should become stationary.

---

## 2. Customer & Order Distributions

### 2.1 LTV Distribution (Gini)

**Confidence: High**

The Pareto principle (80/20 rule) is extensively cited in DTC literature:

- 20% of customers generate ~80% of revenue (widely cited across DTC sources).
- Repeat customers represent 21% of the customer base but generate 44% of total revenue (Mobiloud/industry data).
- Google's CLV research (arxiv:1912.07753) confirms LTV follows a zero-inflated lognormal distribution.
- Elite Shopify merchants (90th percentile) have AOV >$326 vs platform average $78-92, showing power-law concentration.

A Gini coefficient of 0.45-0.80 corresponds to Pareto ratios ranging from 70/30 to 90/10 in terms of top-percentile revenue share.

### 2.2 Order Value CV

**Confidence: Medium**

Direct CV benchmarks for individual order values are scarce:

- Shopify AOV ranges from $78-92 (average) to $326+ (90th percentile), suggesting significant right-skew.
- Shopify experts explicitly note that "averages mask distribution skew" with examples of $600 orders raising means above modal values.
- Weekly revenue CV of 20-37.5% was found (Deloitte, T-shirt example), but individual order CV is wider.

**Gap:** No published source gives explicit order-value CV distributions across DTC brands.

### 2.3 Purchase Interval Distribution

**Confidence: High**

Excellent data from BS&Co analysis of 40,397 repeat buyers:

- 50% repurchase within 30 days, 75% within 90 days.
- 6.3% reorder same day, 15.9% within a week.
- Median time to second purchase: 15-35 days.
- Average time: 50-100+ days (long tail inflates mean).
- Peel Insights and ECPower confirm heavily front-loaded distribution.

This strongly supports a lognormal or similar right-skewed distribution.

### 2.4 Refund Timing Distribution

**Confidence: Medium**

Indirect evidence from return policy data:

- EU mandates 14-day return window (legal minimum).
- Most ecommerce stores offer 15-30 day return windows.
- 72% of customers expect refund credit within 5 days.
- Returns cluster in first 7-14 days after delivery (which is 3-7 days after purchase for most DTC).
- Holiday returns spike 17% above annual average.

**Gap:** No source directly measures the day-by-day distribution of when refunds are initiated relative to purchase date. The 25-65% within 7 days range is an estimate.

---

## 3. Relationships & Correlations

### 3.1 Creative Age vs Performance

**Confidence: High**

Extensive data from Meta advertising ecosystem:

- Meta internal benchmarks: ads >3-4 weeks see 29% higher CPMs, 35% CTR drop.
- Under Andromeda (2024+): creative fatigue in 2-3 weeks vs previous 6-8 weeks.
- 20% week-over-week CTR decline is a standard warning threshold.
- After frequency of 4, CTR drops and CPC rises.
- Recommended refresh cycle: 7-14 days for active spend above $100/day.
- WARC 2025: single-creative campaigns underperform rotating creative by 40%.

### 3.2 Discount vs Repeat

**Confidence: Medium**

Academic and industry research shows mixed effects:

- ScienceDirect: price promotions generate negative long-term effects in repeated-purchase contexts, increasing price sensitivity.
- FasterCapital: initial discount use can drive trial, with reliance diminishing as trust grows.
- Industry consensus: strategic, targeted discounts work better than blanket discounting.
- Overuse trains customers to wait for promotions (the "discount addiction" pattern).

**Caveat:** The correlation magnitude depends heavily on discount frequency and depth. Light discounting (5-10% occasionally) may show near-zero correlation, while heavy discounting (20-30% frequently) shows strongly negative correlation.

### 3.3 Frequency vs Conversion

**Confidence: High**

Clear data from Meta advertising:

- Optimal frequency: 1-2 per user for conversion campaigns.
- Tipping point: 3.4 impressions, after which effectiveness drops.
- At frequency >4: CTR drops, CPC rises.
- Awareness campaigns tolerate higher frequency (2-3/week).
- Average Facebook conversion rate: 8.95% (at optimal frequency).

The inverted-U relationship is well-established but the net correlation in cross-sectional data depends on the frequency distribution.

### 3.4 Refund vs Repeat

**Confidence: Medium**

- 85% of consumers won't purchase again after a difficult return experience.
- Easy return processes can maintain loyalty (hassle-free returns are a competitive advantage).
- The correlation should be negative but the magnitude depends on return experience quality.

### 3.5 Cross-Lag Correlations

**Confidence: Medium**

Marketing mix models typically find cross-lag correlations of 0.2-0.6 between ad spend and downstream outcomes. Meta's attribution windows (7-day click, 1-day view) define the typical lag structure.

---

## 4. Concentration Metrics

### 4.1 Creative Concentration

**Confidence: Medium**

Meta's algorithm creates natural concentration:

- Algorithm prioritizes winning ads, often ignoring new creatives in the same ad set.
- Budget distribution is frequently skewed toward single top performer.
- Meta recommends max 6 creatives per ad set.
- Ad sets with 3-10 creatives lower CPA by 46% vs single creative.
- 70-80% of ad performance stems from creative quality (AppsFlyer 2025).

**Gap:** No published HHI values for creative spend concentration. The range is derived from simulation parameters (30-38 creatives) and the expected Pareto-like distribution.

### 4.2 Customer Concentration

**Confidence: Medium**

With hundreds of customers, HHI is inherently low. The Pareto distribution (20% of customers = 80% of revenue) translates to moderate HHI values in the 0.005-0.04 range for typical DTC customer bases of 500-600.

### 4.3 Channel Asymmetry

**Confidence: Low**

The paid-to-organic ratio is highly variable across DTC brands:

- Early-stage: 60-80% paid revenue (ratio 1.5-4:1).
- Mature: 40-60% paid revenue (ratio 0.6-1.5:1).
- Meta attribution gaps mean true paid influence is underestimated.

**Gap:** No standardized benchmark for paid vs organic revenue split in DTC. This is one of the most variable metrics.

---

## 5. Temporal Evolution

### 5.1 Acquisition Cost Trend

**Confidence: High**

The most well-documented structural trend in DTC:

- Meta CPM: $6.50 (2020) -> $14.90 (2023) -> $17.60 (Q1 2024), ~18% YoY.
- Customer acquisition costs up 222% since 2013.
- 88% of subscription brands report higher YoY acquisition costs.
- DTC founders report 30-70% higher CAC vs 2 years ago.
- Facebook cost per lead: +21% YoY in 2025.
- Structural drivers: Apple privacy changes, auction competition, platform saturation.

### 5.2 Cohort Composition Drift

**Confidence: Medium**

Qualitative pattern confirmed, quantitative benchmarks are scarce:

- Meettie research: newer cohorts repurchase less often and contribute less margin.
- Q4 holiday cohorts show 15-25% lower LTV than Q2 cohorts.
- CAC inflation implies reaching lower-intent audiences over time.
- Degradation is masked by averaging across cohorts.

**Gap:** No published cohort-by-cohort degradation rates across a panel of DTC brands.

### 5.3 Repeat Rate Evolution

**Confidence: Medium**

- Shopify average repeat rate: ~27%.
- DTC brands shifting from acquisition to retention see improving rates.
- Supplement brands improved from 33.1% to 37.7% repurchase rate YoY.
- The trend direction depends heavily on brand strategy and investment.

### 5.4 Trust Baseline Trend

**Confidence: Low**

Trust is a simulator-specific latent variable with no direct external benchmark. Calibration is based on general principles:

- Brands with honest marketing maintain stable trust.
- Promise-experience gaps (common in aggressive Meta advertising) erode trust over time.
- Trust is proxied through NPS, customer satisfaction, and repeat rates externally.

### 5.5 1yr vs 3yr LTV Divergence

**Confidence: Medium**

- Top beauty brands add ~$40 extra LTV by month 12 vs average.
- Median food/beverage brands see customers stop reordering by month 6.
- Retention curves follow logarithmic decay: steep initial drop, then stabilization at 15-25% annual retention.
- Healthy LTV:CAC ratio target is 3:1 after three years.

---

## 6. Causal Mechanisms

### 6.1 & 6.2 Granger Causality (Refund->Trust, Trust->Repeat)

**Confidence: Low**

These are simulator validation checks, not externally benchmarked metrics:

- The causal mechanisms (refund -> trust decline, trust -> repeat purchase) are well-established in academic literature (Frontiers in Psychology 2021, MDPI Systems 2025).
- Granger causality p-values depend on implementation details (lags, sample size, noise).
- The calibrated ranges expect significance at conventional levels if the simulator correctly implements these mechanisms.

### 6.3 Spiral Detection

**Confidence: Low**

No external benchmark exists for "negative spiral episodes." The concept is derived from:

- Known DTC failure mode: quality crisis -> negative reviews -> reduced trust -> declining sales -> cost-cutting -> worse quality.
- In healthy scenarios, mild spirals may occur from seasonal quality dips or supply chain issues.
- Full spirals are rare and typically signal systemic brand problems.

---

## 7. Memory & History Effects

### 7.1 History Effect (Exposure -> Purchase)

**Confidence: Medium**

- Meta attribution data confirms multi-touch influence.
- Optimal frequency data (1-2 impressions) suggests initial positive effect.
- Fatigue at high frequency creates the inverted-U relationship.
- Net positive correlation expected but moderate due to competing effects.

### 7.2 Negative Experience Persistence

**Confidence: Medium**

- 85% of consumers "never" purchase again after bad return experience (for many, effectively permanent).
- NPS detractors take 2-6 months to potentially convert back to promoters.
- 30-day retention after negative experience: 15-25%.
- Half-life of 21-90 days captures the range from quick-recovery to persistent-memory customers.

---

## Key Sources

1. **Syncio** - "$21B in real ecommerce sales" seasonality analysis
2. **BS&Co** - "156K DTC Customers" repeat purchase rate benchmarks
3. **Shopify** - Q4 2024 earnings, merchant statistics, retention guides
4. **Mobiloud** - Repeat customer rate benchmarks (2026)
5. **Rivo** - Shopify Plus retention statistics (2026)
6. **Madgicx** - Meta Ads benchmarks by industry (2025)
7. **WordStream** - Facebook Ads benchmarks (2025)
8. **Upcounting** - Average ecommerce CAC (2025), return rates (2025)
9. **inBeat Agency** - DTC brand statistics (2025)
10. **NestScale** - Facebook ad creative fatigue explained (2025)
11. **DeepSolv** - Meta creative fatigue management (2025)
12. **Deloitte** - Retail Volatility Index research
13. **Google Research** - CLV prediction with zero-inflated lognormal (arxiv:1912.07753)
14. **ScienceDirect** - Impact of price promotions on customer retention
15. **Frontiers in Psychology** - Trust transfer and repeat purchase intention
16. **L.E.K. Consulting** - Fighting rising DTC customer acquisition costs
17. **Focus Digital** - Customer acquisition cost trends (2024)
18. **Meettie/Revenue Roll** - What breaks between acquisition and retention
19. **Propel.ai** - Customer retention rates by industry (2025)
20. **Dollarpocket** - Ecommerce subscription model performance

---

## Patterns Not Covered by Current Checks

The following real-world DTC structural patterns are documented in the research but are not currently measured by the diagnostics checks:

1. **Cart abandonment rate patterns:** Industry average is ~70% cart abandonment. The simulator has cart memory and cart return mechanics but no explicit abandonment rate diagnostic.

2. **Email/SMS attribution:** Most DTC brands derive 20-30% of revenue from email/SMS flows. The simulator focuses on Meta ads and does not model email channel, creating a potential structural gap.

3. **Product return rate by category:** Return rates vary dramatically by category (apparel 30-40%, electronics single digits, beauty low teens). A diagnostic could verify the simulator produces category-appropriate return rates.

4. **Geographic concentration:** DTC brands typically have geographic revenue concentration (major metro areas, specific states/countries). Not currently modeled.

5. **Customer acquisition channel diversification:** Most DTC brands eventually diversify beyond Meta (Google, TikTok, organic social). The simulator is Meta-only, which is realistic for early-stage brands but may underrepresent mature brand dynamics.

6. **Supply chain disruption effects:** Real DTC brands face periodic fulfillment delays, stockouts, and quality issues that create correlated negative shocks. The simulator has fulfillment mechanics but no supply chain disruption diagnostics.

7. **Review and social proof dynamics:** Customer reviews create a feedback loop that affects conversion rates over time. This is not modeled in the simulator.

8. **iOS 14.5+ attribution degradation:** Post-ATT, Meta attribution became noisier. A diagnostic could check whether the simulator's attribution data shows realistic noise levels.

9. **First-order to second-order conversion rate:** The percentage of first-time buyers who make a second purchase is a critical DTC metric (typically 20-30%). Could be a useful standalone diagnostic.

10. **Average days to first repeat purchase by cohort:** Whether this metric is stable or drifting across cohorts would be a useful cohort composition diagnostic.

---

## Caveats and Limitations

1. **Publication bias:** Publicly available benchmarks tend to come from successful or at least surviving brands. The worst-performing brands do not publish their data, creating survivorship bias in the calibration ranges.

2. **Temporal validity:** Benchmarks from 2024-2025 may shift. Meta's algorithm changes (Andromeda update), privacy regulations, and market dynamics can rapidly change structural patterns.

3. **Category specificity:** The calibration targets a generic DTC physical product brand. Specific categories (supplements, fashion, beauty, home goods) have meaningfully different structural patterns.

4. **Scale dependence:** Many structural patterns differ between brands doing $10K/month vs $1M/month in ad spend. The calibration assumes mid-market ($500-$50K monthly ad spend).

5. **Simulator-specific metrics:** Checks 5.4 (trust baseline), 6.1-6.3 (Granger causality, spirals), and 7.2 (negative experience persistence) measure simulator-internal constructs with no direct external benchmark. Their calibration is necessarily speculative.

6. **Missing quantitative data:** For several checks (1.1 autocorrelation, 2.2 order CV, 4.1-4.3 concentration), I could not find published quantitative benchmarks specific to DTC Shopify brands. The calibrated ranges use first-principles reasoning and adjacent data.
