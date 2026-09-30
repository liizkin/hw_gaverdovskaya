# Mobile Phone Price EDA

The dataset contains 2000 rows and 21 columns; price classes are balanced and the associations below are correlations, not evidence of causation.

## Five strongest feature associations

- **ram** (Pearson r = 0.9170): RAM is the clearest basis for price-tier segmentation because devices with more working memory are strongly concentrated in higher tiers.
- **battery_power** (Pearson r = 0.2007): Battery capacity is a useful secondary differentiator because longer-lasting devices tend to occupy higher price tiers.
- **px_width** (Pearson r = 0.1658): Display width resolution can support premium positioning because sharper displays tend to appear in more expensive tiers.
- **px_height** (Pearson r = 0.1489): Display height resolution can support premium positioning because sharper displays tend to appear in more expensive tiers.
- **int_memory** (Pearson r = 0.0444): Internal storage is a modest differentiator because greater built-in capacity is associated with somewhat higher price tiers.

## Data-quality findings

px_height contains 2 zero value(s), which are suspicious because a real phone cannot have zero pixel resolution.

Granularity check: fc has 20 integer-valued levels and is a discrete count, not a continuous measurement.

Granularity check: pc has 21 integer-valued levels and is a discrete count, not a continuous measurement.

Granularity check: clock_speed has 26 recorded levels and is discrete rather than truly continuous; 554 observations are half-steps (including 0.5, 1.5, 2.5).

