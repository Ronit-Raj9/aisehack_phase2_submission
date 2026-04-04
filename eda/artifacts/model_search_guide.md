# Model Search Guide

This guide converts the EDA into a practical experiment search order.

## What The Data Suggests

- The target field lives on a fixed 140x124 grid, so grid-native models remain the first class to try.
- Severe leakage occurs under random overlapping window splits, so validation must be time-safe.
- Train months closest to test by distribution shift: October 2016, April 2016, December 2016, July 2016.
- Most season-sensitive features: v10, NMVOC_finn, rain, u10, cpm25.
- Features most likely to benefit from nonlinear transforms: cpm25, NMVOC_finn, NOx, PM25, SO2, NMVOC_e, NH3, bio.
- Strongest feature-time-series drivers of future PM2.5: cpm25, NMVOC_e, v10, q2, t2, SO2.
- Most redundant channels at the aggregate level: NOx, PM25, NMVOC_finn, SO2, NMVOC_e.
- Sparse spatial priors worth treating specially: NMVOC_e, SO2, NH3, bio, PM25, NMVOC_finn, NOx.
- Features that change the most between clean and extreme PM2.5 regimes: cpm25, t2, pblh, q2, swdown, bio.
- Pairwise relationships that are least season-stable: cpm25 ~ q2, q2 ~ psfc, q2 ~ u10, q2 ~ t2, q2 ~ pblh.
- Features most episode-sensitive: episode masks not available yet.

## Search Order

1. Strong grid baseline: improve the current FNO-style setup with better validation, better normalization, and an episode-aware objective.
2. Local spike specialist: try ConvLSTM or a U-Net-temporal residual head because the horizon is short and episodes are spatially sharp.
3. Hybrid global-local model: combine a spectral/global branch with a local convolutional recurrent branch.
4. Transport-aware variant: if time permits, test graph-style or sparse-edge message passing over the grid or coarse superpixels. This is an inference from the data plus graph forecasting literature, not a direct requirement from the competition.
5. Final ensemble: blend models with complementary behavior, especially one smooth/global model and one spike-sensitive/local model.

## Why These Families

- ConvLSTM was proposed for spatiotemporal sequence forecasting: https://papers.nips.cc/paper_files/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html
- Recent air-quality work continues to explore graph+recurrent hybrids for spatial dependency structure: https://link.springer.com/article/10.1007/s11869-025-01713-8

## Practical Ablations

- Season embedding or month indicator.
- Better train/validation split.
- Raw vs transformed inputs for skewed features.
- Emission channels as static priors vs repeated temporal channels.
- Loss weighting or curriculum focused on episodic regions.

## Recommendation

Start with the strongest reliable baseline you can validate properly, then add only one modeling idea at a time. The EDA says preprocessing, validation design, and episode focus are likely to give better returns than jumping immediately to an exotic architecture.
