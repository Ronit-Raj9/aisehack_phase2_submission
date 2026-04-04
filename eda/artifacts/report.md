# Phase 2 PM2.5 EDA Report

Generated: 2026-04-03T18:44:33

## Problem Lens

Phase 2 is not only a next-step forecasting task. The leaderboard also rewards how well the model tracks spatial structure and magnitude during extreme pollution episodes. That means the analysis should prioritize seasonality, distribution shift, episode rarity, and any preprocessing choice that can suppress sharp spikes.

## Dataset Structure

- April 2016: 715 hourly steps from 2016-04-01 00:00:00 to 2016-04-30 18:00:00 with 690 sliding 26-hour windows at stride 1.
- July 2016: 739 hourly steps from 2016-07-01 00:00:00 to 2016-07-31 18:00:00 with 714 sliding 26-hour windows at stride 1.
- October 2016: 739 hourly steps from 2016-10-01 00:00:00 to 2016-10-31 18:00:00 with 714 sliding 26-hour windows at stride 1.
- December 2016: 739 hourly steps from 2016-12-01 00:00:00 to 2016-12-31 18:00:00 with 714 sliding 26-hour windows at stride 1.
- Test inputs: 218 samples with shape (218, 10, 140, 124). Only 10 lookback hours are available at inference time.

## Most Important Findings

### 1. Seasonal heterogeneity is strong
- December 2016: mean=59.07, sample_p95=215.38, max=1537.67
- October 2016: mean=37.23, sample_p95=107.24, max=2849.73
- April 2016: mean=22.41, sample_p95=58.83, max=2534.45
- July 2016: mean=15.36, sample_p95=45.08, max=843.20

### 2. The provided normalization stats need auditing
- rain: empirical_max=397, provided_max=96.63, ratio=4.109, mean_above_provided_max_frac=0.0000
- cpm25: empirical_max=2850, provided_max=1465, ratio=1.945, mean_above_provided_max_frac=0.0000

### 3. Several features show train-test shift
- u10: KS=0.0803, mean_ratio=0.721, p95_ratio=1.001
- q2: KS=0.0746, mean_ratio=0.926, p95_ratio=1.008
- t2: KS=0.0676, mean_ratio=0.994, p95_ratio=0.997
- psfc: KS=0.0673, mean_ratio=0.998, p95_ratio=1.002
- cpm25: KS=0.0455, mean_ratio=1.048, p95_ratio=0.979
- v10: KS=0.0339, mean_ratio=0.350, p95_ratio=1.082

### 4. Some channels behave like quasi-static seasonal priors
- NMVOC_e: mean relative temporal variability=8.41936e-15
- psfc: mean relative temporal variability=0.00223782
- t2: mean relative temporal variability=0.0111753
- SO2: mean relative temporal variability=0.050335
- NH3: mean relative temporal variability=0.147618
- q2: mean relative temporal variability=0.148491

### 5. Feature relationships to PM2.5
- cpm25: within-month corr with PM2.5 spatial mean=1.000
- rain: within-month corr with PM2.5 spatial mean=-0.029
- NOx: within-month corr with PM2.5 spatial mean=-0.054
- PM25: within-month corr with PM2.5 spatial mean=-0.054
- NH3: within-month corr with PM2.5 spatial mean=-0.061
- SO2: within-month corr with PM2.5 spatial mean=-0.062

### 6. Random overlapping splits are misleading
- April 2016: 1.000 of validation windows overlap at least one train window; mean nearest-window overlap=24.95 hours.
- July 2016: 1.000 of validation windows overlap at least one train window; mean nearest-window overlap=24.88 hours.
- October 2016: 1.000 of validation windows overlap at least one train window; mean nearest-window overlap=24.88 hours.
- December 2016: 1.000 of validation windows overlap at least one train window; mean nearest-window overlap=24.88 hours.

### 7. Episodes are sparse but much more intense
- December 2016: episode_ratio=0.0217, timesteps_with_any_episode=1.0000, episode/non-episode mean ratio=1.85
- April 2016: episode_ratio=0.0208, timesteps_with_any_episode=1.0000, episode/non-episode mean ratio=2.86
- July 2016: episode_ratio=0.0197, timesteps_with_any_episode=1.0000, episode/non-episode mean ratio=3.04
- October 2016: episode_ratio=0.0170, timesteps_with_any_episode=1.0000, episode/non-episode mean ratio=2.80

## Modeling Takeaways

- Use time-blocked or month-aware validation, not a random split over overlapping 26-hour windows.
- Recompute normalization stats from the raw arrays; the provided stats can under-cover the true range.
- Treat emission channels as strong static spatial priors and let dynamic meteorology carry temporal variation.
- Keep architecture and loss choices episode-aware because rare spikes influence two of the three leaderboard terms.
- Expect regime change across seasons and between 2016 train months and 2017 test inputs; month/season conditioning and robust scaling can help.
- Inspect the saved episode hotspot maps before committing to any model that visually over-smooths sharp plumes.

## Output Artifacts

- Tables: `Ronit/eda/artifacts/tables`
- Plots: `Ronit/eda/artifacts/plots`
- Arrays: `Ronit/eda/artifacts/arrays`


## Deep Dive Additions

### Distribution Shape and Normalization
- NMVOC_finn: skewness=48.693, kurtosis=2888.436, mean=4.851e-10, median=0, approx_mode=0, transform=identity_or_log1p, scaler=max_clip_or_robust
- PM25: skewness=41.768, kurtosis=2478.755, mean=3.375e-11, median=5.965e-13, approx_mode=0, transform=identity_or_log1p, scaler=max_clip_or_robust
- NOx: skewness=41.144, kurtosis=2276.946, mean=4.132e-11, median=2.447e-12, approx_mode=0, transform=identity_or_log1p, scaler=max_clip_or_robust
- SO2: skewness=37.245, kurtosis=1819.549, mean=4.25e-11, median=5.102e-13, approx_mode=0, transform=identity_or_log1p, scaler=max_clip_or_robust
- rain: skewness=35.165, kurtosis=2142.859, mean=0.08605, median=0, approx_mode=0, transform=identity, scaler=robust
- NH3: skewness=26.367, kurtosis=1256.382, mean=3.088e-11, median=6.858e-12, approx_mode=0, transform=identity_or_log1p, scaler=max_clip_or_robust
- NMVOC_e: skewness=25.524, kurtosis=1048.659, mean=4.927e-11, median=1.344e-12, approx_mode=0, transform=identity_or_log1p, scaler=max_clip_or_robust
- bio: skewness=8.094, kurtosis=89.679, mean=5.024e-11, median=0, approx_mode=0, transform=identity_or_log1p, scaler=max_clip_or_robust

### Outlier-Heaviness
- rain: upper-outlier-frac=0.2346, max/p99=28.657, zero-frac=0.705
- bio: upper-outlier-frac=0.2292, max/p99=4.373, zero-frac=1.000
- SO2: upper-outlier-frac=0.1391, max/p99=117.618, zero-frac=0.999
- PM25: upper-outlier-frac=0.1308, max/p99=67.127, zero-frac=1.000
- NMVOC_e: upper-outlier-frac=0.1281, max/p99=16.665, zero-frac=1.000
- NOx: upper-outlier-frac=0.1233, max/p99=49.436, zero-frac=1.000
- cpm25: upper-outlier-frac=0.0912, max/p99=4.473, zero-frac=0.000
- pblh: upper-outlier-frac=0.0728, max/p99=1.643, zero-frac=0.000

### Temporal Behavior
- swdown: lag1=0.942, lag24=0.997, diurnal_amplitude=738
- bio: lag1=0.939, lag24=0.997, diurnal_amplitude=1.836e-10
- t2: lag1=0.960, lag24=0.993, diurnal_amplitude=7.346
- pblh: lag1=0.957, lag24=0.993, diurnal_amplitude=879.4
- q2: lag1=0.989, lag24=0.934, diurnal_amplitude=0.0006976
- SO2: lag1=0.941, lag24=0.920, diurnal_amplitude=2.163e-12
- NH3: lag1=0.941, lag24=0.911, diurnal_amplitude=5.731e-12
- u10: lag1=0.981, lag24=0.870, diurnal_amplitude=1.248

### Spatial Concentration
- SO2: mean top1pct mass share=0.727, mean spatial CV=14.788
- NMVOC_finn: mean top1pct mass share=0.723, mean spatial CV=12.026
- NOx: mean top1pct mass share=0.426, mean spatial CV=7.574
- PM25: mean top1pct mass share=0.328, mean spatial CV=5.826
- NMVOC_e: mean top1pct mass share=0.200, mean spatial CV=3.555
- u10: mean top1pct mass share=0.172, mean spatial CV=5.561
- rain: mean top1pct mass share=0.166, mean spatial CV=2.417
- bio: mean top1pct mass share=0.139, mean spatial CV=2.220

### Which Train Month Looks Closest to Test?
- October 2016: mean KS=0.0557, median KS=0.0514, mean Wasserstein=27.9
- April 2016: mean KS=0.0727, median KS=0.0475, mean Wasserstein=50.73
- December 2016: mean KS=0.0910, median KS=0.0975, mean Wasserstein=35.32
- July 2016: mean KS=0.1369, median KS=0.1126, mean Wasserstein=67.74

### Forecastability Ranking
- cpm25: mean |corr| to future PM2.5 over 0-16h = 0.940, best lead = 0h
- NMVOC_e: mean |corr| to future PM2.5 over 0-16h = 0.823, best lead = 13h
- v10: mean |corr| to future PM2.5 over 0-16h = 0.790, best lead = 13h
- q2: mean |corr| to future PM2.5 over 0-16h = 0.761, best lead = 0h
- t2: mean |corr| to future PM2.5 over 0-16h = 0.747, best lead = 1h
- SO2: mean |corr| to future PM2.5 over 0-16h = 0.700, best lead = 13h
- psfc: mean |corr| to future PM2.5 over 0-16h = 0.698, best lead = 21h
- u10: mean |corr| to future PM2.5 over 0-16h = 0.651, best lead = 24h

### Redundancy / Multicollinearity
- NOx: VIF=249.35, max |corr with another feature|=0.983
- PM25: VIF=226.58, max |corr with another feature|=0.980
- NMVOC_finn: VIF=129.04, max |corr with another feature|=0.983
- SO2: VIF=120.57, max |corr with another feature|=0.840
- NMVOC_e: VIF=75.54, max |corr with another feature|=0.840
- NH3: VIF=69.90, max |corr with another feature|=0.926
- swdown: VIF=37.29, max |corr with another feature|=0.973
- bio: VIF=36.19, max |corr with another feature|=0.973

### Regime Sensitivity
- cpm25: mean |extreme-clean|=3.464 std, mean |same-time spearman|=1.000, regime monotonicity=0.983
- t2: mean |extreme-clean|=1.164 std, mean |same-time spearman|=0.547, regime monotonicity=-0.820
- pblh: mean |extreme-clean|=1.047 std, mean |same-time spearman|=0.484, regime monotonicity=-0.813
- q2: mean |extreme-clean|=1.031 std, mean |same-time spearman|=0.488, regime monotonicity=-0.262
- swdown: mean |extreme-clean|=0.927 std, mean |same-time spearman|=0.390, regime monotonicity=-0.902
- bio: mean |extreme-clean|=0.924 std, mean |same-time spearman|=0.451, regime monotonicity=-0.883
- v10: mean |extreme-clean|=0.850 std, mean |same-time spearman|=0.314, regime monotonicity=0.143
- psfc: mean |extreme-clean|=0.821 std, mean |same-time spearman|=0.162, regime monotonicity=-0.728

### Pairwise Relationship Stability
- cpm25 ~ q2: corr std across months=0.483, range=1.168, sign flip=1
- q2 ~ psfc: corr std across months=0.441, range=1.057, sign flip=1
- q2 ~ u10: corr std across months=0.436, range=1.105, sign flip=1
- q2 ~ t2: corr std across months=0.399, range=0.989, sign flip=1
- q2 ~ pblh: corr std across months=0.343, range=0.937, sign flip=1
- q2 ~ v10: corr std across months=0.318, range=0.831, sign flip=0
- cpm25 ~ v10: corr std across months=0.318, range=0.806, sign flip=1
- q2 ~ NH3: corr std across months=0.284, range=0.705, sign flip=0

### Seasonality Severity
- v10: seasonal mean ratio=-1.597, seasonal mean CV=6.724
- NMVOC_finn: seasonal mean ratio=322.484, seasonal mean CV=1.491
- rain: seasonal mean ratio=8.772, seasonal mean CV=1.168
- u10: seasonal mean ratio=15.686, seasonal mean CV=0.667
- cpm25: seasonal mean ratio=3.847, seasonal mean CV=0.577
- PM25: seasonal mean ratio=2.917, seasonal mean CV=0.502
- bio: seasonal mean ratio=2.747, seasonal mean CV=0.369
- NMVOC_e: seasonal mean ratio=1.913, seasonal mean CV=0.340

### Normalization Blueprint
- NH3: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 1.974e-10], role=static_sparse_prior
- NMVOC_e: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 4.869e-10], role=static_sparse_prior
- NMVOC_finn: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 1.192e-09], role=static_sparse_prior
- NOx: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 4.743e-10], role=static_sparse_prior
- PM25: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 3.315e-10], role=static_sparse_prior
- SO2: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 3.619e-10], role=static_sparse_prior
- bio: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 9.795e-10], role=static_sparse_prior
- cpm25: recipe=box_cox -> winsorize_to_p01_p99 -> center=median -> scale=iqr, clip=[0.0521, 262.4079], role=autoregressive_anchor

### Role Recommendations
- cpm25: role=autoregressive_anchor, transform=box_cox, scaler=robust, future-PM2.5 score=0.940
- v10: role=highly_dynamic_driver, transform=identity, scaler=robust, future-PM2.5 score=0.790
- u10: role=highly_dynamic_driver, transform=identity, scaler=robust, future-PM2.5 score=0.651
- rain: role=highly_dynamic_driver, transform=identity, scaler=robust, future-PM2.5 score=0.152
- q2: role=mixed_signal, transform=identity, scaler=standardize, future-PM2.5 score=0.761
- psfc: role=mixed_signal, transform=identity, scaler=standardize, future-PM2.5 score=0.698
- t2: role=stable_diurnal_driver, transform=identity, scaler=standardize, future-PM2.5 score=0.747
- pblh: role=stable_diurnal_driver, transform=identity, scaler=robust, future-PM2.5 score=0.260
- swdown: role=stable_diurnal_driver, transform=identity, scaler=standardize, future-PM2.5 score=0.116
- NMVOC_e: role=static_sparse_prior, transform=identity_or_log1p, scaler=max_clip_or_robust, future-PM2.5 score=0.823

### Episode Context
- December 2016: episode ratio=0.0217, episode PM2.5 p95=351.53, max episodic grid points in one hour=1432
- October 2016: episode ratio=0.0170, episode PM2.5 p95=299.45, max episodic grid points in one hour=986
- April 2016: episode ratio=0.0208, episode PM2.5 p95=216.05, max episodic grid points in one hour=1375
- July 2016: episode ratio=0.0197, episode PM2.5 p95=153.80, max episodic grid points in one hour=1661

### Artifact Expansion
- Every CSV table is also rendered as paginated PNG images under `plots/table_images`.
- Each feature now gets its own diagnostic figure under `plots/feature_diagnostics`.
- Additional guides are saved as `preprocessing_guide.md` and `model_search_guide.md`.
- Added regime-response, pair-stability, normalization-blueprint, and episode-signature diagnostics.
