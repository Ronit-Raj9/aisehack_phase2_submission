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
