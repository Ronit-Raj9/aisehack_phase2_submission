# Winning Strategy for ANRF AISEHack Phase 2 PM2.5 Forecasting

## 1. What The Competition Actually Rewards

This competition is not just "predict the next PM2.5 map well on average."

From the problem statement and helper description:

- Input at inference time is only `10` historical hours for every feature.
- Output must be `16` future PM2.5 maps with shape `(140, 124, 16)`.
- Evaluation is a hidden weighted average of:
  - normalized `GlobalSMAPE`
  - normalized `EpisodeCorr`
  - normalized `EpisodeSMAPE`
- Episode masks are defined from the ground-truth target using a monthly STL decomposition:
  - residual spike over local baseline
  - `residual > 3 * residual_std`
  - PM2.5 > 1

This has very important modeling consequences:

- A model that is smooth and globally decent can still lose badly if it misses sharp episode structure.
- A model that only chases spikes can lose on global SMAPE.
- The best model is the one that is balanced:
  - low global magnitude error
  - good spatial pattern alignment in high-PM regions
  - good amplitude accuracy in episodic pixels

In other words: this is a short-horizon spatiotemporal forecasting problem with hidden multi-objective scoring and explicit emphasis on extreme-event fidelity.

## 2. What The EDA Says Matters Most

After reading the full EDA reports, tables, and core plots, these are the highest-value conclusions.

### 2.1 Seasonality is strong and real

The four train months are not interchangeable.

- `December 2016` has the highest mean PM2.5 and strongest sustained winter pollution.
- `October 2016` is also heavily polluted and is the closest train month to test by feature-distribution shift.
- `July 2016` is the farthest from test overall and behaves differently because of monsoon dynamics.

Implication:

- You should not trust a random split.
- You should tune on season-aware validation.
- You should expect models to fail if they overfit one month regime.

### 2.2 Random sliding-window validation is badly misleading

The EDA showed essentially complete leakage under the current sample construction logic:

- all validation windows overlap train windows
- nearest overlap is about `25` shared hours out of `26`

This means the current baseline validation in [prepare_dataset.py](/home/raj/Documents/CODING/Hackathon/ANRF_AISEHack_Final_2/Ronit/scripts/prepare_dataset.py) is not trustworthy.

Implication:

- Fixing validation is the first improvement, before changing architecture.

### 2.3 The current provided min-max statistics are not safe enough

EDA found the provided stats under-cover at least:

- `rain`
- `cpm25`

Implication:

- Do not rely on the provided `feat_min_max.mat` for final preprocessing.
- Recompute train-only statistics on the full 2016 raw arrays.

### 2.4 Train-test shift is real

The largest train-test shifts are in:

- `u10`
- `q2`
- `t2`
- `psfc`
- `cpm25`
- `v10`

And the closest month to test is:

- `October 2016`, then `April 2016`

Implication:

- Models must be robust to modest regime change.
- Validation should include an October-like stress test.

### 2.5 Emission channels are mostly sparse spatial priors, not rich temporal signals

The EDA consistently shows:

- `NMVOC_e`, `SO2`, `NH3`, `PM25`, `NOx`, `NMVOC_finn`, `bio` are extremely sparse
- most have almost no meaningful temporal variability compared with meteorological channels
- several are highly redundant

But one subtle point matters:

- some emission channels still have strong future-PM2.5 association through spatial structure
- especially `NMVOC_e`

Implication:

- Do not drop emission channels blindly.
- Do not feed them as if they were equally dynamic 10-step sequences.
- Use them as static or slowly varying spatial priors in a dedicated branch.

### 2.6 PM2.5 episodes are sparse in space, but present at every hour

The EDA episode summary is very revealing:

- episode ratio is only about `1.7%` to `2.2%` of grid cells
- but `timesteps_with_any_episode_frac = 1.0` for every month

That means:

- every hour has at least some episodic pixels
- episode metrics matter on every forecast horizon
- the problem is not "detect rare event hours"
- the problem is "find and size sparse extreme subregions at all times"

This is one of the most important insights in the whole project.

### 2.7 Spatial structure matters a lot

The PM2.5 spatial maps and hotspot plots clearly show:

- stable high-pollution corridors over the Indo-Gangetic Plain / northern belt
- strong regional structure in October and December
- emissions are highly concentrated spatially
- sharp plumes exist and over-smoothing will hurt episodic metrics

Implication:

- pure per-pixel time-series models are not enough
- your model must preserve sharp local structure while keeping global context

### 2.8 The most useful dynamic drivers are not all equally stable

Feature roles from the EDA:

- `cpm25`: autoregressive anchor
- `v10`, `u10`, `rain`: highly dynamic drivers
- `t2`, `pblh`, `swdown`: stable diurnal drivers
- `q2`, `psfc`: mixed but useful
- emissions: sparse static priors

The pairwise-stability analysis also showed:

- `q2` has some of the least season-stable relationships
- several `q2` pairings even flip sign across months

Implication:

- do not build simplistic global linear assumptions around humidity
- let the model learn regime-dependent interactions

### 2.9 Step-by-step visual read of the main plots

After going back through the full plot folders, the most important composite figures are:

- [distribution_heatmaps.png](artifacts/plots/distribution_heatmaps.png)
- [skew_transform_and_month_similarity.png](artifacts/plots/skew_transform_and_month_similarity.png)
- [shift_and_staticness.png](artifacts/plots/shift_and_staticness.png)
- [feature_role_map.png](artifacts/plots/feature_role_map.png)
- [cpm25_spatial_maps.png](artifacts/plots/cpm25_spatial_maps.png)
- [episode_spatial_panels.png](artifacts/plots/episode_spatial_panels.png)
- [episode_hotspot_geography.png](artifacts/plots/episode_hotspot_geography.png)
- [pairwise_correlation_stability.png](artifacts/plots/pairwise_correlation_stability.png)
- [target_regime_feature_response.png](artifacts/plots/target_regime_feature_response.png)

Reading them in order gives a very clear story.

1. [distribution_heatmaps.png](artifacts/plots/distribution_heatmaps.png) shows that the dynamic meteorology channels are not all equally difficult:
   - `q2`, `t2`, `u10`, `v10`, and `psfc` have low to moderate skew and are not the main normalization problem.
   - `cpm25` is consistently right-skewed in every month.
   - `rain` is the hardest dynamic variable statistically because it is both zero-inflated and extremely heavy-tailed.
   - emission channels have near-total zero domination and extreme kurtosis, which means they are not "continuous meteorology-like" variables at all.
2. [skew_transform_and_month_similarity.png](artifacts/plots/skew_transform_and_month_similarity.png) confirms two things:
   - `cpm25` is the only feature where a real nonlinear transform is clearly justified by the visuals.
   - `October 2016` is the closest train month to test for many of the important channels, while `April 2016` is especially relevant for `q2` and `v10`.
3. [shift_and_staticness.png](artifacts/plots/shift_and_staticness.png) cleanly separates "shift-sensitive" from "static-like":
   - biggest shift: `u10`, `q2`, `t2`, `psfc`, `cpm25`
   - nearly static despite predictive value: `NMVOC_e`
   - extremely dynamic: `rain`, `u10`, `v10`
4. [feature_role_map.png](artifacts/plots/feature_role_map.png) visually justifies using separate branches:
   - `cpm25` sits alone as the autoregressive anchor
   - `u10`, `v10`, and `rain` are dynamic drivers
   - `t2`, `pblh`, and `swdown` are structured diurnal drivers
   - emissions cluster as static sparse priors with very different sparsity from meteorology
5. [cpm25_spatial_maps.png](artifacts/plots/cpm25_spatial_maps.png) is one of the most important figures in the project:
   - `October` and `December` have a broad northern high-PM corridor and larger spatially coherent polluted zones.
   - `July` contracts into a different, monsoon-shaped regime with much lower mean PM2.5.
   - high-PM frequency maps show that extreme PM is not random pixel noise; it follows geographic structure.
6. [episode_spatial_panels.png](artifacts/plots/episode_spatial_panels.png) shows something subtle but crucial:
   - episode frequency exists across the map, but residual standard deviation is much stronger in specific regions, especially winter northern belts.
   - the model therefore needs both broad coverage and region-dependent amplitude control.
7. [episode_hotspot_geography.png](artifacts/plots/episode_hotspot_geography.png) shows that hotspot locations are recurring but not perfectly fixed across months.
   - this is another reason not to rely on a memorized static mask alone.
   - a good model should use static priors to bias the search space, then let meteorology move and reshape the plume.
8. [pairwise_correlation_stability.png](artifacts/plots/pairwise_correlation_stability.png) reinforces that some relationships are not season-invariant.
   - `q2` is involved in several unstable pairings.
   - feature engineering should therefore help the network express interactions, but the final model should remain nonlinear and regime-aware.
9. [target_regime_feature_response.png](artifacts/plots/target_regime_feature_response.png) confirms that extreme PM2.5 is most sensitive to:
   - `cpm25`
   - `t2`
   - `pblh`
   - `q2`
   - `swdown`
   - `bio`

This visual sequence strongly supports a model that is:

- direct multi-horizon, not recursive
- split into dynamic and static branches
- built to preserve sharp local structure
- trained with season-aware validation
- explicitly biased toward episode-sensitive regions

### 2.10 What the feature-diagnostic images show, feature by feature

The 16 feature panels under [feature_diagnostics](artifacts/plots/feature_diagnostics) add very practical preprocessing guidance.

- [cpm25_diagnostic.png](artifacts/plots/feature_diagnostics/cpm25_diagnostic.png)
  - Mean is much larger than median and mode in every month, which is the classic heavy-right-tail signature.
  - The temporal profile is strongly persistent and the future-lead correlations dominate every other feature.
  - October is visually closest to test and December is the hardest high-pollution regime.
  - Interpretation: this feature deserves privileged model capacity, a nonlinear input transform, robust scaling, and explicit recent-trend features.
- [q2_diagnostic.png](artifacts/plots/feature_diagnostics/q2_diagnostic.png)
  - Mean and median are relatively close, so the problem is not gross skewness.
  - The same-time and lead relations with PM2.5 are clearly informative, but the direction and strength vary by month.
  - Interpretation: standardize it, keep it dynamic, and let it interact with `t2`, `pblh`, and wind instead of assuming a globally fixed monotone effect.
- [t2_diagnostic.png](artifacts/plots/feature_diagnostics/t2_diagnostic.png)
  - Mean and median are close, with only mild left skew.
  - The daily cycle is very clean and the relation with future PM2.5 is stably negative.
  - Interpretation: this is a strong regime/diurnal conditioning signal and should be kept in the main dynamic branch with simple standardization.
- [u10_diagnostic.png](artifacts/plots/feature_diagnostics/u10_diagnostic.png)
  - The distribution is signed, moderately tailed, and test-shifted.
  - Mean, median, and mode are not the issue; physically meaningful sign and outliers are.
  - Interpretation: preserve sign, use robust scaling, and derive wind speed/direction features.
- [v10_diagnostic.png](artifacts/plots/feature_diagnostics/v10_diagnostic.png)
  - This channel is also signed and dynamic, but the month-to-month pattern is even less stable than `u10`.
  - Some correlations change strength and the closest train month to test is not always the same as for other variables.
  - Interpretation: keep raw signed `v10`, use robust scaling, and do not collapse wind information to speed alone.
- [swdown_diagnostic.png](artifacts/plots/feature_diagnostics/swdown_diagnostic.png)
  - The plots are visually bimodal because nights are near zero and days are high.
  - Mean alone is misleading here; the day-night structure matters more than Gaussianity.
  - Interpretation: use the raw channel, standardize it, and add a daylight mask because this variable is effectively an implicit clock.
- [pblh_diagnostic.png](artifacts/plots/feature_diagnostics/pblh_diagnostic.png)
  - Positive with a long upper tail and a very strong diurnal structure.
  - Future PM2.5 tends to be lower when PBLH is high, consistent with mixing.
  - Interpretation: robust scaling is better than standard scaling, and interactions with wind or inverse-PBLH are worthwhile.
- [psfc_diagnostic.png](artifacts/plots/feature_diagnostics/psfc_diagnostic.png)
  - Low variability, low seasonal CV, and near-standardizable behavior.
  - It still carries signal, but it is a secondary continuous conditioning feature rather than a dominant driver.
  - Interpretation: standardize it and keep it as a stable background meteorological context variable.
- [rain_diagnostic.png](artifacts/plots/feature_diagnostics/rain_diagnostic.png)
  - This is the clearest mean-median-mode mismatch after PM2.5.
  - Median and mode are effectively zero while the mean is driven by rare high spikes.
  - Interpretation: do not try to force this into a Gaussian-like channel. Split it into occurrence and amount, then scale the amount robustly.
- [NMVOC_e_diagnostic.png](artifacts/plots/feature_diagnostics/NMVOC_e_diagnostic.png)
  - Almost flat temporally, extremely sparse spatially, yet strongly associated with future PM2.5.
  - This is a textbook static-prior feature.
  - Interpretation: keep it, but move it to a dedicated static encoder instead of spending temporal capacity on repeated copies.
- [SO2_diagnostic.png](artifacts/plots/feature_diagnostics/SO2_diagnostic.png)
  - Also almost static with extremely concentrated mass.
  - The feature is useful through geography, not through short-term time variation.
  - Interpretation: static branch, sparse-preserving scaling, and optional top-tail clipping.
- [NH3_diagnostic.png](artifacts/plots/feature_diagnostics/NH3_diagnostic.png)
  - Very sparse, weaker than `SO2` and `NMVOC_e`, but still informative enough to retain.
  - Interpretation: keep as a low-capacity static prior.
- [PM25_diagnostic.png](artifacts/plots/feature_diagnostics/PM25_diagnostic.png)
  - Sparse and high-kurtosis with limited dynamic content.
  - Interpretation: static prior or low-priority channel; drop only if memory becomes a real bottleneck.
- [NOx_diagnostic.png](artifacts/plots/feature_diagnostics/NOx_diagnostic.png)
  - Similar story to `PM25`, with strong sparsity and redundancy.
  - Interpretation: keep for the stronger models; prune only in small ablations.
- [NMVOC_finn_diagnostic.png](artifacts/plots/feature_diagnostics/NMVOC_finn_diagnostic.png)
  - Extremely sparse and also highly seasonal.
  - Interpretation: useful as seasonal context, but regularize its branch because it can encourage month memorization.
- [bio_diagnostic.png](artifacts/plots/feature_diagnostics/bio_diagnostic.png)
  - Sparse, seasonal, and more regime-sensitive than its raw values first suggest.
  - Interpretation: good candidate for the static branch plus episode-focused fine-tuning because it becomes more relevant in polluted regimes.

### 2.11 What the transform previews really mean

The transform previews under [transform_previews](artifacts/plots/transform_previews) are very useful because they show where normalization effort is worth spending.

- [cpm25_transform_preview.png](artifacts/plots/transform_previews/cpm25_transform_preview.png)
  - This is the only feature where the visual gain from `log1p` or Box-Cox is obvious and substantial.
  - Box-Cox looks better than `log1p` for making the bulk of the distribution more symmetric.
  - Recommendation: use Box-Cox if your training pipeline can support stable inverse transforms; otherwise use `log1p` as the simpler fallback.
- Emission transform previews:
  - [NMVOC_e_transform_preview.png](artifacts/plots/transform_previews/NMVOC_e_transform_preview.png)
  - [SO2_transform_preview.png](artifacts/plots/transform_previews/SO2_transform_preview.png)
  - [NH3_transform_preview.png](artifacts/plots/transform_previews/NH3_transform_preview.png)
  - [NOx_transform_preview.png](artifacts/plots/transform_previews/NOx_transform_preview.png)
  - [PM25_transform_preview.png](artifacts/plots/transform_previews/PM25_transform_preview.png)
  - [NMVOC_finn_transform_preview.png](artifacts/plots/transform_previews/NMVOC_finn_transform_preview.png)
  - [bio_transform_preview.png](artifacts/plots/transform_previews/bio_transform_preview.png)
  - In these figures, `log1p` changes the tail a little, but it does not solve the core structure because the core structure is the point mass at zero.
  - Recommendation: preserve sparsity, clip the extreme top tail, and spend modeling effort on using these maps as priors rather than trying to Gaussianize them.
- `rain`
  - There is no transform preview for rain because the correct move is not "pick the best monotone transform."
  - The correct move is to represent rain as two variables: event occurrence and positive intensity.
- `swdown`
  - The issue is not skew; it is bimodality from day versus night.
  - A daylight indicator is more useful than any nonlinear transform.

The transform previews therefore imply three normalization families, not one:

- `cpm25`: nonlinear transform plus robust scaling
- meteorology with continuous signed or near-symmetric behavior: identity plus robust or standard scaling
- sparse/static emissions: sparse-preserving clip-and-scale, not aggressive warping

### 2.12 What the images imply for the most generalized model

The image review makes the generalization recipe more concrete.

- Separate "where pollution tends to exist" from "what moves and intensifies it":
  - emissions and geography define a prior support
  - `cpm25` and meteorology define the moving state
- Treat seasonality as a domain shift problem:
  - October-like validation is mandatory
  - December is the winter stress test
  - July should be kept in training because it teaches the model not to confuse monsoon-clean with low-confidence predictions
- Learn both amplitude and edges:
  - global context alone will over-smooth
  - local convolutions alone will miss corridor-scale transport
- Do not chase a single universal normalization rule:
  - the plots very clearly show that `cpm25`, `rain`, and sparse emissions need different treatments
- Keep the model direct multi-horizon:
  - lead-correlation plots stay informative across the full horizon
  - recursive rollouts would just accumulate blur and bias
- Add one auxiliary task tied to extremes:
  - future episode mask
  - future top-decile PM mask
  - or future anomaly/residual map

This is why the best generalized solution is not "pick the fanciest backbone." It is "use the right data treatment, the right validation, and the right hybrid inductive bias."

### 2.13 Step-by-step read of the major tables

The table outputs under [tables](artifacts/tables) are dense, but they are extremely useful once grouped by purpose. This is the cleanest way to read them.

#### Validation and leakage tables

- [window_leakage_months.csv](artifacts/tables/window_leakage_months.csv)
- [window_overlap_by_offset.csv](artifacts/tables/window_overlap_by_offset.csv)

What they say:

- Every random validation split built from stride-1 `26`-hour windows is heavily contaminated.
- The mean nearest-window overlap is about `24.9` hours out of `26`, which is almost a duplicate-sample situation.

What this means for modeling:

- Any architecture comparison done on the original random split is suspect.
- A simpler model with clean blocked validation is more trustworthy than a fancier model tuned on leaked validation.

#### Shift tables

- [train_test_shift.csv](artifacts/tables/train_test_shift.csv)
- [monthwise_train_test_shift.csv](artifacts/tables/monthwise_train_test_shift.csv)
- [monthwise_train_test_shift_summary.csv](artifacts/tables/monthwise_train_test_shift_summary.csv)

What they say:

- `October 2016` is the closest month to test overall.
- `April 2016` is the next closest and is especially relevant for `q2` and `v10`.
- `July 2016` is the farthest train month from test.
- The biggest feature-level shifts are in `u10`, `q2`, `t2`, `psfc`, and `cpm25`.

What this means for modeling:

- October-holdout validation is the closest offline proxy to leaderboard behavior.
- A generalized model should not be tuned only on winter-heavy folds.
- Wind and humidity features need robust handling because they move between regimes even when PM2.5 itself is only moderately shifted.

#### Distribution and normalization tables

- [feature_distribution_diagnostics.csv](artifacts/tables/feature_distribution_diagnostics.csv)
- [feature_transform_recommendations.csv](artifacts/tables/feature_transform_recommendations.csv)
- [feature_normalization_blueprint.csv](artifacts/tables/feature_normalization_blueprint.csv)
- [normalization_audit.csv](artifacts/tables/normalization_audit.csv)

What they say:

- `cpm25` is strongly right-skewed and is the main continuous variable that clearly benefits from a nonlinear transform.
- `rain` is not just skewed; it is structurally zero-inflated with rare large bursts.
- Emission channels are mostly point-mass-at-zero variables with extreme kurtosis, so "normalizing them like weather variables" is the wrong mental model.
- The provided normalization bounds are not safe enough for final work:
  - `rain` empirical max is about `397` versus provided max `96.6`
  - `cpm25` empirical max is about `2850` versus provided max `1465`
  - provided `cpm25` minimum also sits above a nontrivial part of the real train distribution

What this means for modeling:

- Recompute stats from raw train data.
- Keep a different normalization family for PM2.5, dynamic meteorology, and sparse priors.
- Use clipping and robust scaling to stabilize training, but avoid hard-truncating the target values used for metric computation.

#### Temporal profile tables

- [cpm25_hourly_profile.csv](artifacts/tables/cpm25_hourly_profile.csv)
- [cpm25_autocorrelation.csv](artifacts/tables/cpm25_autocorrelation.csv)
- [feature_temporal_diagnostics.csv](artifacts/tables/feature_temporal_diagnostics.csv)
- [episode_hourly_profile.csv](artifacts/tables/episode_hourly_profile.csv)

What they say:

- PM2.5 has a strong diurnal structure in every month, with nighttime and late-evening values higher than late-morning values.
- The diurnal amplitude varies a lot by month:
  - July about `2.57`
  - April about `4.86`
  - October about `8.58`
  - December about `19.77`
- PM2.5 spatial-mean autocorrelation is extremely high:
  - lag `1`: `0.9976`
  - lag `12`: `0.9058`
  - lag `24`: about `0.977`
- Episode ratio is also hourly structured:
  - highest around `00:00` to `02:00`
  - lowest around `07:00` to `10:00`
  - rises again toward late evening
- `swdown`, `t2`, and `pblh` carry the cleanest diurnal signals.
- `rain` has low persistence and behaves more like an intermittent event flag than a smooth state variable.

What this means for modeling:

- Time-of-day information matters even if the competition does not hand you an explicit hour feature.
- `swdown` is especially valuable because it gives the model an implicit daylight clock.
- Horizon conditioning is justified because the response at `t+1` and `t+16` is not identical, even though persistence stays strong.

#### Spatial and episode tables

- [feature_spatial_diagnostics.csv](artifacts/tables/feature_spatial_diagnostics.csv)
- [cpm25_spatial_summary.csv](artifacts/tables/cpm25_spatial_summary.csv)
- [episode_summary.csv](artifacts/tables/episode_summary.csv)
- [episode_hotspots.csv](artifacts/tables/episode_hotspots.csv)

What they say:

- PM2.5 is spatially concentrated but not ultra-sparse, with top-`1%` cells carrying around `4.5%` to `5%` of total PM2.5 mass.
- Several emissions are much more concentrated:
  - `SO2` top-`1%` mass share around `0.73`
  - `NMVOC_finn` top-`1%` mass share around `0.72`
  - `NOx` top-`1%` mass share around `0.43`
- Episodes are sparse in area but continuous in time:
  - episode ratio roughly `1.7%` to `2.2%`
  - timesteps with any episode = `1.0` in every month
- December has the highest residual-variability regime and the highest episodic PM intensity.

What this means for modeling:

- The model must localize sparse extreme regions every hour, not just detect whether an episode hour exists.
- Static emission priors are most useful as location bias, not as dynamic sequence signals.
- A purely smooth decoder will underperform because the scoring emphasizes sparse high-intensity structure.

#### Forecastability and target-association tables

- [feature_forecastability_summary.csv](artifacts/tables/feature_forecastability_summary.csv)
- [feature_lead_correlations.csv](artifacts/tables/feature_lead_correlations.csv)
- [feature_corr_with_cpm25.csv](artifacts/tables/feature_corr_with_cpm25.csv)
- [feature_target_regime_summary.csv](artifacts/tables/feature_target_regime_summary.csv)

What they say:

- Best short-horizon forecastability:
  - `cpm25` by far
  - then `NMVOC_e`, `v10`, `q2`, `t2`, `SO2`, `psfc`, `u10`
- But this needs careful interpretation:
  - `NMVOC_e` is highly forecastive because it is a strong static spatial prior, not because it carries short-term dynamics.
  - pooled feature-to-target correlations can be misleading because they mix season and regime effects.
- The difference between pooled and within-month correlation is especially important:
  - `psfc` is strongly positive in pooled correlation with PM2.5, but slightly negative within month
  - `u10` and `v10` look much stronger in pooled correlation than within-month correlation
  - this means some of the apparent signal is seasonal confounding, not purely hourly causal influence
- Regime tables show that the strongest clean-to-extreme shifts are in:
  - `cpm25`
  - `t2`
  - `pblh`
  - `q2`
  - `swdown`
  - `bio`

What this means for modeling:

- Use `cpm25` as the anchor state.
- Use wind, humidity, temperature, and boundary-layer features as nonlinear conditioning variables, not as single monotone controls.
- Keep one auxiliary head or weighting strategy focused on high-PM or episodic regions because the regime tables show that important features change meaningfully between clean and extreme states.

#### Redundancy and stability tables

- [feature_redundancy_vif.csv](artifacts/tables/feature_redundancy_vif.csv)
- [feature_pair_stability_summary.csv](artifacts/tables/feature_pair_stability_summary.csv)
- [feature_correlation_matrix.csv](artifacts/tables/feature_correlation_matrix.csv)
- [feature_pair_monthly_correlation.csv](artifacts/tables/feature_pair_monthly_correlation.csv)

What they say:

- Emission variables are heavily redundant with each other.
- Even some meteorological variables have large VIF because shared seasonality and diurnal structure create strong collinearity.
- `q2` is central to the least stable pairs:
  - `cpm25 ~ q2`
  - `q2 ~ psfc`
  - `q2 ~ u10`
  - `q2 ~ t2`
  - `q2 ~ pblh`

What this means for modeling:

- Linear feature selection based only on correlation or VIF is not enough.
- A neural model should keep most variables, but branch them sensibly:
  - dynamic state branch
  - static sparse prior branch
  - optional derived-interaction channels
- If memory becomes an issue, compress redundant emissions before dropping meteorology.

### 2.14 Table-driven synthesis for the most generalized model

Once you combine the image review with the table outputs, the generalized-model recipe becomes very specific.

1. What the model should memorize:
   - stable geography
   - broad pollution corridors
   - static emission priors
2. What the model should infer online from the last `10` hours:
   - current PM2.5 state
   - wind-driven transport tendency
   - boundary-layer ventilation regime
   - daylight versus nighttime regime
   - precipitation washout risk
3. What the model should not assume:
   - one fixed humidity effect
   - one fixed wind effect
   - a single seasonal normalization
   - that forecastive static channels are dynamic causal drivers
4. What the training process should explicitly stress:
   - October-like shift
   - December-like severe episodes
   - nighttime and late-evening high-episode conditions
   - sparse hotspot fidelity

That is why the strongest generalized model is a hybrid that learns:

- geography through a static prior branch
- fast-changing atmosphere through a dynamic branch
- both corridor-scale and plume-scale structure through global and local spatial modules
- episode-sensitive refinement through auxiliary supervision or weighted loss

## 3. What Is Wrong In The Current Baseline Pipeline

Reading the repo code confirms that the current baseline leaves several high-value improvements on the table.

### 3.1 Data leakage in validation

[prepare_dataset.py](/home/raj/Documents/CODING/Hackathon/ANRF_AISEHack_Final_2/Ronit/scripts/prepare_dataset.py) does a random split after creating heavily overlapping 26-hour windows.

That is exactly the leakage pattern your EDA proved is invalid.

### 3.2 Provided min-max stats are used directly

[prepare_dataset.py](/home/raj/Documents/CODING/Hackathon/ANRF_AISEHack_Final_2/Ronit/scripts/prepare_dataset.py) and [infer.py](/home/raj/Documents/CODING/Hackathon/ANRF_AISEHack_Final_2/Ronit/scripts/infer.py) use the provided `feat_min_max.mat`.

The EDA already showed that these stats are not enough for final-quality preprocessing.

### 3.3 Loss is not aligned to the actual leaderboard

[train.py](/home/raj/Documents/CODING/Hackathon/ANRF_AISEHack_Final_2/Ronit/scripts/train.py) uses an `LpLoss`.

But the leaderboard is driven by:

- SMAPE
- episode SMAPE
- episode correlation

So even a better offline `L2` can still be the wrong direction for the competition.

### 3.4 The current FNO2D input design is very simple

[baseline_model.py](/home/raj/Documents/CODING/Hackathon/ANRF_AISEHack_Final_2/Ronit/models/baseline_model.py) flattens `time * features` into channels and applies 2D spectral blocks.

That is a solid baseline, but it:

- does not explicitly model temporal order beyond channel position
- does not separate static and dynamic channels
- does not explicitly target episode structure
- is vulnerable to over-smoothing local spikes

## 4. Non-Negotiable Fixes Before Fancy Model Work

These are the first things I would change before any major architecture jump.

### 4.1 Build time-safe validation

Recommended validation design:

1. Month-holdout validation:
   - primary fold: hold out `October 2016`
   - stress fold: hold out `December 2016`
2. Within-month blocked validation:
   - use the last `96` to `144` hours of each month as validation
   - add a `26`-hour exclusion gap before validation starts
3. Episode-focused score slice:
   - separately report score on windows whose future target has large episode ratio or large PM2.5 p95

Why this is best:

- October is the closest train month to test
- December stresses winter episodes
- blocked tails measure leakage-free generalization
- the episode slice stops you from optimizing only average conditions

### 4.2 Use the exact competition metric offline

For every validation run:

- compute future episode masks using the exact helper-notebook STL logic on the validation targets
- compute:
  - `GlobalSMAPE`
  - `EpisodeSMAPE`
  - `EpisodeCorr`
- normalize them exactly like the competition
- choose checkpoints by a balanced surrogate, not by `L2`

Recommended offline selection score:

`offline_score = 0.34 * NormGlobalSMAPE + 0.33 * NormEpisodeSMAPE + 0.33 * NormEpisodeCorr`

Because hidden leaderboard weights are unknown, keep the surrogate balanced.

I also recommend logging:

- each component separately
- the minimum of the three components

That helps avoid a model that wins on one component and collapses on another.

### 4.3 Fit preprocessing on the full train data, not sampled train stats

For final modeling, compute transformation/scaling stats on all 2016 train pixels, not just samples.

Fit on:

- all four train months
- training split only
- full raw arrays
- chunked passes if memory is tight

Persist:

- clip thresholds
- transform parameters
- centering stats
- scaling stats
- target transform parameters

### 4.4 Use only the allowed 10-hour history for every feature

Do not let training accidentally benefit from future feature hours.

The competition explicitly removed future exogenous features at inference time.

### 4.5 Make predictions nonnegative

PM2.5 is nonnegative and negative outputs are harmful.

Use one of:

- `softplus` output head
- predict transformed target and invert with positivity preserved
- final clamp to `>= 0`

## 5. Complete Preprocessing Plan For All Features

This is the preprocessing plan I would actually use for complete data.

### 5.1 Global preprocessing rules

- Fit all preprocessing on training data only.
- Compute train-wide robust statistics on the full raw data.
- Keep physical sign information for `u10` and `v10`.
- Do not spatially flip or rotate India maps.
- Use separate treatment for dynamic channels and static-sparse channels.
- Store preprocessing parameters with the model and inference notebook.

### 5.2 Feature-by-feature preprocessing

| Feature | Role | How to use it | Transform | Scaling / clipping | Extra engineered features | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| `cpm25` | autoregressive anchor | full 10-step temporal input and target basis | Box-Cox or `log1p` for model input; keep original-scale path for loss | robust center/scale; clip input side around train `p01`-`p99` or `p99.5`; do not hard-clip labels | lag-1 delta, lag-3 delta, 3h mean, 6h mean, rolling std, last-minus-mean | Most important channel. Give it privileged capacity. |
| `q2` | mixed but strong | full 10-step temporal input | identity | mean/std | interactions with `t2`, `pblh`; temporal derivative | Useful, but seasonally unstable. Let the network learn nonlinear interactions. |
| `t2` | stable diurnal driver | full 10-step temporal input | identity | mean/std | temporal derivative, anomaly from last-10 mean | Very useful during extreme regimes and highly forecastive. |
| `u10` | dynamic driver | full 10-step temporal input | identity | robust scale in physical sign-preserving units | wind speed, direction cosine, advection proxy | Shifted at test time; robust scaling matters. |
| `v10` | dynamic driver | full 10-step temporal input | identity | robust scale in physical sign-preserving units | wind speed, direction sine, advection proxy | One of the strongest dynamic drivers. |
| `swdown` | stable diurnal driver | full 10-step temporal input | identity | mean/std | daylight mask (`swdown > 0`), derivative | Helps infer hour-like regime even if timestamps are unavailable. |
| `pblh` | stable diurnal driver | full 10-step temporal input | identity | robust scale; clip high tails | inverse PBLH, stagnation interaction with wind/rain | Important for trapping vs mixing. |
| `psfc` | mixed, low-variance | full 10-step temporal input | identity | mean/std | anomaly from recent mean | Stable but still useful. |
| `rain` | intermittent scavenging driver | full 10-step temporal input | identity or `log1p` on positive-only sub-branch | robust scale; clip top tail | rain mask, clipped rain amount, consecutive-rain indicator | Best treated as two channels: occurrence and amount. |
| `PM25` | static sparse prior | static branch, not main dynamic branch | `log1p`-style sparse-preserving transform | clip to train `p99`; scale by clipped upper stat | none or optional local neighborhood pooling | Keep as static prior. |
| `NH3` | static sparse prior | static branch | `log1p`-style sparse-preserving transform | clip to train `p99`; sparse-preserving scale | none | Keep, but do not spend dynamic capacity on it. |
| `SO2` | static sparse prior | static branch | `log1p`-style sparse-preserving transform | clip to train `p99`; sparse-preserving scale | none | Strongly concentrated spatial prior. |
| `NOx` | static sparse prior | static branch or low-priority channel | `log1p`-style sparse-preserving transform | clip to train `p99`; sparse-preserving scale | none | Highly redundant; drop only if memory is tight. |
| `NMVOC_e` | static sparse prior with strong value | static branch | `log1p`-style sparse-preserving transform | clip to train `p99`; sparse-preserving scale | none | Keep for sure; high forecastability despite near-static behavior. |
| `NMVOC_finn` | sparse seasonal prior | static branch, lower trust | `log1p`-style sparse-preserving transform | clip to train `p99`; sparse-preserving scale | none | Highly seasonal and sparse; use but regularize. |
| `bio` | sparse prior / seasonal interaction | static branch or weak dynamic branch | `log1p`-style sparse-preserving transform | clip to train `p99`; sparse-preserving scale | none | Useful as seasonal biomass-burning context. |

### 5.2.1 How mean, median, and mode should guide your normalization

This dataset is a good example of why you should look at mean, median, and mode together instead of relying on a single skewness score.

- If `mean >> median >> mode`, the channel is tail-dominated:
  - `cpm25` is the cleanest example
  - use a nonlinear transform or a residual-style target design
- If `median = mode = 0` and the mean is nonzero because of rare spikes, the channel is zero-inflated rather than merely skewed:
  - `rain`
  - all emission maps
  - use either a split representation or a static sparse branch
- If `mean ≈ median` and skewness is modest, simple scaling is enough:
  - `q2`
  - `t2`
  - `psfc`
- If the feature is signed and physically directional, preserve sign and use robust scaling instead of nonlinear warping:
  - `u10`
  - `v10`

In practice, I would save the following full-train summary for every feature and use it in experimentation:

- mean
- median
- approximate mode
- standard deviation
- MAD or IQR
- `p01`, `p05`, `p95`, `p99`, `p99.5`
- zero fraction
- outlier-high fraction

That gives you enough information to choose transformations rationally instead of by habit.

### 5.2.2 Exact normalization families I would implement

I would not use one global scaler object for all variables. I would implement three separate feature families.

1. PM2.5 family
   - feature: `cpm25`
   - transform: Box-Cox on the full train values, or `log1p` fallback
   - scaling: robust center and scale on transformed values
   - clipping: soft input clipping near train `p99` or `p99.5`, never hard-clipping targets used for scoring
2. Continuous meteorology family
   - `q2`, `t2`, `psfc`, `swdown`: standardize
   - `u10`, `v10`, `pblh`, `rain_amount`: robust-scale
   - preserve sign for wind
   - add binary indicators where the variable is structurally mixed:
     - `rain_mask`
     - `daylight_mask`
3. Sparse prior family
   - features: all emissions
   - transform: identity or very light `log1p`
   - clipping: upper-tail clipping using train `p99` or `p99.5`
   - scale: divide by clipped upper statistic or use a robust sparse-aware scale
   - pass through static encoder, not 10 repeated dynamic frames

If you implement only one preprocessing upgrade from this whole document, make it this separation. It is one of the highest-leverage changes available.

### 5.2.3 Exact initial clip windows and scaling targets I would start from

The blueprint table already gives concrete defaults. If I were building the first serious training pipeline, I would start from these train-derived values and only relax them if ablations show harm.

- `cpm25`
  - transform: Box-Cox
  - clip window before robust scaling: about `[0.052, 262.4]`
  - center: median
  - scale: IQR
- `rain_amount`
  - keep a separate binary `rain_mask`
  - clip positive amount at about `1.93` before robust scaling
  - center: median
  - scale: IQR
- `pblh`
  - clip window about `[71.9, 3184.2]`
  - robust center/scale
- `u10`
  - keep signed values
  - clip window about `[-13.65, 18.94]`
  - robust center/scale
- `v10`
  - keep signed values
  - clip window about `[-16.29, 22.23]`
  - robust center/scale
- `q2`
  - no special nonlinear transform
  - standardize on raw values
  - upper train range is about `0.0247`, so large test drift beyond this should be watched
- `t2`
  - standardize on raw values
  - practical train support is about `[238.1, 319.6]`
- `swdown`
  - standardize on raw values
  - keep zeros
  - add `daylight_mask = 1(swdown > 0)`
- `psfc`
  - standardize on raw values
  - practical train range is about `[48935.7, 101872.1]`
- sparse emission family
  - preserve zeros
  - center at median
  - scale by clipped upper statistic
  - approximate upper clips from the blueprint:
    - `NH3`: `1.97e-10`
    - `NMVOC_e`: `4.87e-10`
    - `NMVOC_finn`: `1.19e-09`
    - `NOx`: `4.74e-10`
    - `PM25`: `3.32e-10`
    - `SO2`: `3.62e-10`
    - `bio`: `9.80e-10`

Important caution:

- These values are strong defaults, not sacred constants.
- Refit them on the complete train arrays in your final preprocessing script.
- Do not recompute them on validation or test.

### 5.2.4 What not to misread from the tables

A few table results are easy to misuse.

- High forecastability does not always mean a channel belongs in the temporal branch.
  - `NMVOC_e` is the classic example.
- Large pooled correlation does not always mean large within-regime effect.
  - `psfc`, `u10`, and `v10` illustrate this clearly.
- High VIF does not mean "drop this variable" in a nonlinear model.
  - It often means "avoid wasting capacity by encoding redundant channels separately at full width."
- Very high zero fraction in sparse emissions does not mean "useless."
  - It often means "location prior with strong spatial meaning."

So the right lesson is not aggressive feature pruning. The right lesson is better architecture and preprocessing structure.

### 5.3 Additional static channels to add

I strongly recommend adding:

- normalized row index
- normalized column index
- latitude
- longitude

Why:

- the baseline only uses generic `[0, 1]` grid coordinates
- actual lat/lon carries meteorologically meaningful structure
- India has strong north-south and west-east gradients

### 5.4 Derived features worth testing

These are high-value engineered features that are physically plausible and compatible with the EDA.

#### Wind-derived

- `wind_speed = sqrt(u10^2 + v10^2)`
- `wind_dir_cos = u10 / (wind_speed + eps)`
- `wind_dir_sin = v10 / (wind_speed + eps)`

Why:

- transport matters
- raw `u10` and `v10` sign changes are useful, but speed/direction often make learning easier

#### PM2.5 dynamics

- lag-1 difference
- lag-3 difference
- last-step anomaly from recent 3h or 6h mean
- rolling standard deviation over the 10-hour lookback

Why:

- episodes are relative spikes over local baseline
- recent acceleration often matters more than absolute level alone

#### Stagnation / scavenging proxies

- `stagnation = low_wind * low_pblh`
- `washout = rain_mask * rain_amount`
- `ventilation_proxy = wind_speed * pblh`

Why:

- PM2.5 episodes often strengthen under stagnant, shallow boundary layers
- rain suppresses PM2.5 through wet deposition

#### Optional advection proxy

Use the latest PM2.5 map and approximately shift it using local wind direction as a coarse one-step transport proxy.

Why:

- the horizon is short
- advection dominates some short-term spatial movement

This is a physics-inspired feature, not a full physics model.

### 5.5 How to handle emissions correctly

This is one of the most important preprocessing choices.

Do this:

- treat emissions as static priors
- either take the last lookback frame only, or average over the 10 lookback frames
- process them through a separate encoder
- fuse them later with dynamic meteorology and PM2.5 features

Do not do this:

- spend equal temporal modeling capacity on all 10 repeated emission frames
- assume they should be normalized the same way as dynamic meteorology

### 5.6 Target preprocessing

For the target, I recommend one of these two strategies:

#### Strategy A: transformed target training + original-scale loss

- model predicts transformed PM2.5 internally
- invert to original scale before the main competition-aligned losses

Why this is strong:

- stabilizes heavy-tailed target training
- still aligns losses with the original metric

#### Strategy B: dual-head base + residual target

- Head 1 predicts a smooth base field
- Head 2 predicts a positive residual / spike correction
- final output = `base + residual`

Why this is strong:

- the competition episode definition is explicitly "baseline + spike"
- this makes the architecture closer to the scoring logic

If I had to choose one, I would prioritize Strategy B for the best model and Strategy A for the safest baseline upgrade.

## 6. Sample Construction And Weighting

### 6.1 Window creation

Use:

- input = first `10` hours
- target = next `16` hours
- stride `1` for the final run

If compute is tight during prototyping:

- pretrain with stride `2` or `3`
- finetune with stride `1`

### 6.2 Balanced sampling across months

Avoid letting the easiest month dominate.

Recommended:

- month-balanced batches
- or month-level weights so every month contributes similarly

### 6.3 Episode-aware sample weighting

This is a major opportunity.

Build per-sample weights from the future target, for example:

- future episode ratio
- future PM2.5 `p95`
- future PM2.5 `max`

Example weighting idea:

`sample_weight = 1 + a * future_episode_ratio + b * normalized_future_p95`

This helps the model see more hard windows without collapsing into only-extreme training.

### 6.4 Multi-horizon weighting

Because the leaderboard averages all 16 horizons, keep horizon weights mostly balanced.

A good starting point:

- uniform horizon weights

Then test:

- slightly larger weights on later horizons `12-16`

Why:

- later steps are harder
- but over-weighting them can damage early-step calibration

## 7. Model Families: What To Use And What Not To Start With

## 7.1 Best first family: corrected FNO2D / AFNO-style spectral model

Why it fits this problem:

- full-grid prediction
- global spatial coupling
- short horizon
- weather-like gridded dynamics

Why it fits the EDA:

- PM2.5 has broad regional structure
- long-range context matters
- October/December pollution corridors are large-scale as well as local

Why this family is credible:

- FNO introduced efficient operator learning for PDE-like systems: `https://arxiv.org/abs/2010.08895`
- FourCastNet showed Fourier/operator-style models can work very well in fast weather forecasting: `https://authors.library.caltech.edu/records/k959a-53q45`

What to change relative to the repo baseline:

- fix preprocessing
- fix validation
- replace plain `LpLoss`
- add temporal stem instead of naive channel flattening only
- add a local residual head or sharper decoder

Verdict:

- This should be one of your main models.

## 7.2 Best local spike family: U-Net temporal model or ConvLSTM hybrid

Why it fits this problem:

- episodes are spatially sparse and sharp
- local plume edges matter
- 10-hour input history is short enough for temporal convolution or ConvLSTM to work well

Why this family is credible:

- U-Net remains the standard strong local encoder-decoder family: `https://arxiv.org/abs/1505.04597`
- ConvLSTM was designed for spatiotemporal frame forecasting: `https://researchportal.hkust.edu.hk/en/publications/convolutional-lstm-network-a-machine-learning-approach-for-precip`

How to use it here:

- temporal encoder over the 10-hour stack
- full-resolution U-Net decoder
- direct 16-horizon head
- optional residual correction on top of a smoother global model

Verdict:

- This should be your second main family.

## 7.3 Best overall single-model direction: hybrid global-local model

If you want the strongest single-model direction, I would not choose plain FNO2D or plain U-Net alone.

I would choose:

- global spectral branch
- local convolutional branch
- static emission branch
- direct multi-horizon output
- auxiliary episode head

Why:

- spectral branch helps global transport-like structure
- local branch preserves sharp plumes
- static branch uses sparse emission priors correctly
- auxiliary episode head aligns training with leaderboard emphasis

This is the model I would bet on as the best single architecture direction.

## 7.4 Graph / mesh / transport-aware models

Why they are attractive:

- transport is directional
- spatial dependency is not purely isotropic
- graph methods can encode nonlocal edges and wind-informed connectivity

Why they are credible:

- Graph WaveNet is a strong general spatiotemporal graph baseline: `https://www.ijcai.org/proceedings/2019/264`
- GraphCast shows graph/mesh methods can be excellent for geophysical forecasting: `https://doi.org/10.1126/science.adi2336`
- PM2.5-specific STGNN work also reports gains, especially for harder rare/high-concentration prediction: `https://www.sciencedirect.com/science/article/pii/S095965262303038X`

Why I would not start here:

- your domain is a dense fixed grid, not sparse stations
- a full 17k-node graph is heavy
- implementation risk is much higher
- Kaggle notebook constraints matter

Best use of this family:

- optional coarse-graph branch
- superpixel graph
- graph residual corrector
- not your first full-production model

Verdict:

- Worth trying only after you have a strong FNO/U-Net baseline.

## 7.5 Transformers

Transformers are not wrong for this problem, but I would not make them your first bet.

Why:

- sequence length is short (`10` input hours)
- data volume is limited (four months only)
- full-grid space-time attention is expensive
- over-smoothing or overfitting is easy

Why they are still relevant:

- AirFormer shows transformer variants can work very well in air-quality forecasting: `https://doi.org/10.1609/aaai.v37i12.26676`

Best way to use them here:

- only after spatial downsampling or patchification
- or as a horizon decoder on top of a conv/spectral spatial encoder

Verdict:

- second-tier option, not first-tier.

## 7.6 FNO3D

This is the most tempting architecture that I would explicitly rank behind improved FNO2D + hybrid models.

Why it is tempting:

- joint space-time operator learning
- elegant treatment of spatiotemporal volumes

Why I would not start with it:

- more memory-heavy
- higher overfitting risk on only four train months
- easier to produce smooth forecasts that hurt episode metrics
- less flexible for separating static vs dynamic channels

Verdict:

- try it only after a strong corrected FNO2D baseline exists.

## 7.7 Full physics-informed PINNs

I do not recommend starting with a full PINN here.

Why:

- the task is supervised short-term forecasting on gridded simulated data
- a full PDE-constrained training setup is much heavier and riskier
- the competition reward is forecast score, not PDE residual

Why physics still matters:

- physics-inspired features are valuable
- transport-aware inductive bias is valuable
- advection/stagnation proxies are valuable

PINN background if you want to explore later:

- `https://doi.org/10.1016/j.jcp.2018.10.045`

Verdict:

- use physics-inspired design, not full PINN-first design.

## 8. My Recommended Best Model

If I were building the best single model for this competition, I would build this.

### 8.1 Architecture

#### Inputs

- dynamic branch:
  - past `cpm25`
  - `q2`, `t2`, `u10`, `v10`, `swdown`, `pblh`, `psfc`, `rain`
  - derived wind and PM2.5 dynamics features
- static branch:
  - all emission maps
  - latitude / longitude / row / col

#### Dynamic encoder

Use either:

- shallow 3D temporal-conv stem
- or ConvLSTM stem

Goal:

- encode short-horizon temporal order explicitly before spatial forecasting

#### Global branch

- AFNO / FNO2D-style blocks on encoded dynamic features
- broad receptive field
- smoother large-scale transport structure

#### Local branch

- U-Net or residual conv encoder-decoder
- preserves local edges and plume intensity
- can be conditioned on the same dynamic embedding plus static emissions

#### Static prior branch

- 1x1 conv or shallow CNN over static emission maps + geolocation
- produce a prior feature bank
- fuse into both global and local branches

#### Fusion

- concatenate branch features
- use channel attention / squeeze-excitation / gated fusion

#### Output head

- direct `16`-step prediction head
- one shared decoder with horizon embeddings, or grouped horizon heads
- positive output enforcement with `softplus` or post-inverse clamp

#### Auxiliary head

Add at least one auxiliary training target:

- future episode mask classification
- or future high-PM mask classification
- or future residual-anomaly map

Why this is valuable:

- it directly trains the model to care about sparse episodic regions
- it improves EpisodeCorr and EpisodeSMAPE without abandoning the regression target

### 8.2 Why this is the best match to your EDA

- `cpm25` is the strongest anchor -> dynamic branch must privilege it
- emissions are sparse priors -> static branch is correct
- episodes are sparse but always present -> auxiliary sparse-event head is useful
- spatial corridors are broad and structured -> global spectral branch helps
- local spikes matter -> local U-Net / ConvLSTM branch helps

### 8.3 Training details that matter specifically for generalization

The image review sharpens a few training choices that are easy to miss.

- Use horizon conditioning.
  - The lead-correlation plots stay informative across all `16` steps, but the best predictor mix changes with horizon.
  - Give the decoder a horizon embedding or grouped horizon heads instead of forcing one identical mapping for all steps.
- Prefer GroupNorm or LayerNorm over BatchNorm in the main body.
  - The train months are heterogeneous and the test month is shifted.
  - BatchNorm can leak mini-batch regime composition into the features in an unhelpful way.
- Train in two phases.
  - Phase 1: balanced broad training for overall stability.
  - Phase 2: fine-tune with stronger episode or high-PM weighting.
  - This matches the visual evidence that broad regional structure and local extreme fidelity are both required.
- Use season-balanced mini-batches.
  - Do not let December dominate only because its PM levels are larger.
  - A generalized model should see all regimes regularly.
- Use EMA for checkpoint smoothing.
  - The target is noisy and the objective is multi-term.
  - EMA often improves robustness at inference for exactly this kind of spatiotemporal forecasting task.
- Keep regularization moderate, not extreme.
  - recommended starting range:
    - dropout: `0.05` to `0.15`
    - weight decay: `1e-4` to `5e-4`
    - stochastic depth: `0.0` to `0.1` if used
  - Too much regularization will blur the very edges the episode metrics care about.
- Use direct multi-output prediction, not autoregressive rollout.
  - The spatial plots show how much damage over-smoothing could do.
  - Recursive prediction compounds that risk.

### 8.4 Capacity allocation suggested by the tables

The tables also suggest how to spend model capacity.

- Give the dynamic branch most of the width.
  - `cpm25`, wind, `q2`, `t2`, `pblh`, and `swdown` deserve most of the representation budget.
- Keep the static branch narrower but persistent.
  - Emissions are valuable, but many are redundant.
  - A shallow encoder with a moderate channel count is usually enough.
- Avoid giving all inputs equal channel budget.
  - `cpm25` should have privileged access either through extra channels, residual paths, or special engineered features.
- Use the static prior branch at multiple scales.
  - one fusion at coarse resolution for corridor bias
  - one fusion at finer resolution for hotspot localization
- Keep the output head expressive enough to model horizon differences.
  - grouped horizon heads or horizon embeddings are worth the extra parameters
- Reserve some learning signal for episode structure.
  - an auxiliary mask head or weighted residual head is more valuable here than marginally increasing backbone depth

If forced to choose between a deeper backbone and better branch design, I would choose better branch design for this competition.

## 9. Training Objective

The loss should reflect the hidden multi-objective leaderboard, not just RMSE.

### 9.1 Recommended main loss

A good starting hybrid loss is:

`L = 0.35 * Huber(original_or_inverted_scale) + 0.25 * SMAPE_all + 0.20 * SMAPE_episode_weighted + 0.10 * corr_loss_on_episode_like_pixels + 0.10 * gradient_or_edge_loss`

Meaning:

- `Huber`: stable regression backbone
- `SMAPE_all`: aligns with global metric
- `SMAPE_episode_weighted`: emphasizes hard pixels
- `corr_loss_on_episode_like_pixels`: helps EpisodeCorr
- `gradient_or_edge_loss`: discourages over-smoothing

### 9.2 Practical simpler version

If you want a safer first implementation:

`L = 0.5 * Huber + 0.3 * SMAPE_all + 0.2 * episode_weighted_SMAPE`

Then add correlation loss only if validation shows it helps.

### 9.3 How to build the episode weight map

Best option:

- exact future episode masks from the helper-notebook STL logic on train targets

Simpler option:

- soft weights based on future PM2.5 percentile or residual z-score

Example:

- weight `1.0` for normal pixels
- weight `2.0` to `4.0` for episodic pixels

Do not make the weight too large; global SMAPE still matters.

## 10. Hyperparameters Worth Searching

These are the parameters that are actually worth your time.

### 10.1 Data and preprocessing

| Parameter | Good range / options | Why it matters |
| --- | --- | --- |
| validation split | Oct-holdout, Dec-holdout, blocked-tail | biggest source of false progress |
| target transform | none, `log1p`, Box-Cox, base+residual | target is heavy-tailed |
| cpm25 input clip | `p99`, `p99.5`, none | stabilize inputs without killing extremes |
| rain handling | raw, mask+amount, `log1p` positive-only | intermittent and heavy-tailed |
| emission handling | full dynamic, static branch, averaged static branch | huge effect on efficiency and inductive bias |
| derived features | none vs wind-speed/deltas/stagnation | often high-return additions |

### 10.2 FNO / AFNO branch

| Parameter | Suggested search |
| --- | --- |
| width | `48, 64, 80, 96` |
| modes | `12, 16, 20, 24` |
| number of spectral blocks | `3-6` |
| dropout | `0.0-0.10` |
| channel lifting width | `1x-2x base width` |

### 10.3 U-Net / Conv branch

| Parameter | Suggested search |
| --- | --- |
| base channels | `32, 48, 64` |
| depth | `3-5` levels |
| temporal kernel | `3 or 5` |
| dilations | `1, 2, 4` |
| normalization | GroupNorm or LayerNorm |

### 10.4 Optimization

| Parameter | Suggested range |
| --- | --- |
| optimizer | `AdamW` |
| learning rate | `1e-4` to `2e-3` |
| weight decay | `1e-5` to `5e-4` |
| batch size | as large as memory allows, usually `2-8` for bigger models |
| scheduler | cosine decay or one-cycle; step LR is acceptable but not optimal |
| gradient clip | `1.0` or `5.0` |
| mixed precision | yes |
| EMA | optional but useful |

### 10.5 Sampling / weighting

| Parameter | Suggested search |
| --- | --- |
| month balancing | on / off |
| episode oversampling factor | `1.5x, 2x, 3x` |
| horizon weights | uniform vs slightly tail-heavier |
| auxiliary episode head weight | `0.05-0.25` |

## 11. Ensembling: Should You Use It?

Yes, but only a small, diverse ensemble.

### 11.1 Why ensemble here

Because the leaderboard combines:

- global magnitude accuracy
- episodic magnitude accuracy
- episodic spatial correlation

Different architectures fail differently:

- FNO-like models often do global smooth structure well
- U-Net / ConvLSTM-style models often preserve local sharpness better
- different losses may trade off global vs episode behavior

That is exactly the situation where ensembling helps.

### 11.2 What ensemble to use

Best practical ensemble:

1. corrected FNO2D / AFNO-like model
2. local U-Net-temporal or ConvLSTM hybrid
3. optional third model:
   - another seed
   - another loss weighting
   - another validation-chosen checkpoint style

### 11.3 How to blend

- blend on original PM2.5 scale after inverse transforms
- clamp to nonnegative after blending
- choose weights using the offline competition surrogate

Good starting weights:

- `0.45` global spectral model
- `0.35` local sharpness model
- `0.20` alternate seed or residual model

If you only have two models:

- start with `0.55 / 0.45`
- then optimize on offline score

### 11.4 What not to do with ensembles

- do not ensemble five nearly identical seeds and call it diversity
- do not average transformed-space outputs directly
- do not pick weights using only public leaderboard movement

## 12. Suggested Experiment Roadmap

If your time is limited, this is the order I would follow.

### Stage A: fix the baseline first

- leakage-free validation
- train-only full-data normalization
- nonnegative outputs
- offline exact competition metric

This alone can materially improve your real score.

### Stage B: metric-aligned corrected FNO2D

- same family as baseline
- better preprocessing
- better loss
- better checkpoint selection

This gives a clean strong baseline.

### Stage C: local sharpness model

- temporal U-Net or ConvLSTM/U-Net hybrid
- same corrected preprocessing
- same offline scoring

This tells you whether local models recover episode structure better.

### Stage D: best hybrid model

- spectral global branch
- local branch
- static emission branch
- episode auxiliary head

This is the best single-model target.

### Stage E: final small ensemble

- only after offline validation clearly supports it

## 13. Things I Would Explicitly Avoid

- Random train/validation split over sliding windows.
- Blind use of the provided min-max file for final runs.
- Pure `L2` / `Lp` optimization as your main selection criterion.
- Heavy reliance on timestamps unless you confirm they are available at inference time.
- Spatial flips or rotations as augmentation.
- Starting with full PINNs.
- Starting with full-grid transformers before you have a strong conv/spectral baseline.
- Treating emission channels as fully dynamic 10-step sequences.
- Overfitting to the public leaderboard with many small unvalidated changes.

## 14. Final Recommendation In One Paragraph

To maximize your chance of winning, first fix the dataset pipeline and evaluation pipeline, because the current baseline has leakage and weak metric alignment. Then build a corrected FNO2D-style full-grid model with full-data train-only preprocessing, robust nonnegative target handling, and a hybrid loss aligned to GlobalSMAPE plus episode accuracy. After that, build a second model that is sharper locally, preferably a U-Net-temporal or ConvLSTM hybrid with a separate static-emission branch. The strongest final direction is a fused global-local model with an auxiliary episode head, followed by a small diverse ensemble of that model with a corrected spectral baseline. Focus more on validation design, feature handling, target/loss design, and episode-aware training than on jumping immediately to the most exotic architecture.

## 15. Research References Used For This Strategy

- Competition problem statement and metric: local repo files and screenshots.
- Fourier Neural Operator: `https://arxiv.org/abs/2010.08895`
- U-Net: `https://arxiv.org/abs/1505.04597`
- ConvLSTM: `https://researchportal.hkust.edu.hk/en/publications/convolutional-lstm-network-a-machine-learning-approach-for-precip`
- Graph WaveNet: `https://www.ijcai.org/proceedings/2019/264`
- FourCastNet / adaptive Fourier operators for weather forecasting: `https://authors.library.caltech.edu/records/k959a-53q45`
- GraphCast / graph-based geophysical forecasting: `https://doi.org/10.1126/science.adi2336`
- AirFormer for air-quality forecasting: `https://doi.org/10.1609/aaai.v37i12.26676`
- PM2.5-specific STGNN evidence: `https://www.sciencedirect.com/science/article/pii/S095965262303038X`
- PINNs background: `https://doi.org/10.1016/j.jcp.2018.10.045`
