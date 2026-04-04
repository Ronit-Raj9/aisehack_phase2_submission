# Preprocessing Guide

This guide is generated from the EDA and is intended to help with normalization and transform choices.

## Global Rules

- Fit all transforms and scalers on the training split only to avoid leakage.
- Prefer time-safe validation splits before fitting normalization statistics.
- For strongly right-skewed positive variables, consider `log1p` or Box-Cox style warping.
- For sign-changing skewed variables, prefer a Yeo-Johnson style transform.
- For heavy outliers, robust centering/scaling is often safer than standardization alone.

## Feature-Level Recommendations

- NH3: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 1.974e-10], skew=26.367, zero_frac=1.000, seasonal_mean_cv=0.168, reason=Highly sparse emission-like field; preserve zeros and avoid over-warping the spatial prior.
- NMVOC_e: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 4.869e-10], skew=25.524, zero_frac=1.000, seasonal_mean_cv=0.340, reason=Highly sparse emission-like field; preserve zeros and avoid over-warping the spatial prior.
- NMVOC_finn: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 1.192e-09], skew=48.693, zero_frac=0.995, seasonal_mean_cv=1.491, reason=Highly sparse emission-like field; preserve zeros and avoid over-warping the spatial prior.
- NOx: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 4.743e-10], skew=41.144, zero_frac=1.000, seasonal_mean_cv=0.156, reason=Highly sparse emission-like field; preserve zeros and avoid over-warping the spatial prior.
- PM25: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 3.315e-10], skew=41.768, zero_frac=1.000, seasonal_mean_cv=0.502, reason=Highly sparse emission-like field; preserve zeros and avoid over-warping the spatial prior.
- SO2: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 3.619e-10], skew=37.245, zero_frac=0.999, seasonal_mean_cv=0.080, reason=Highly sparse emission-like field; preserve zeros and avoid over-warping the spatial prior.
- bio: recipe=identity_or_log1p -> winsorize_to_p01_p99 -> center=median -> scale=p99_after_clip, clip=[0, 9.795e-10], skew=8.094, zero_frac=1.000, seasonal_mean_cv=0.369, reason=Highly sparse emission-like field; preserve zeros and avoid over-warping the spatial prior.
- cpm25: recipe=box_cox -> winsorize_to_p01_p99 -> center=median -> scale=iqr, clip=[0.0521, 262.4079], skew=4.696, zero_frac=0.000, seasonal_mean_cv=0.577, reason=Strictly positive and strongly right-skewed.
- pblh: recipe=identity -> winsorize_to_p01_p99 -> center=median -> scale=iqr, clip=[71.8993, 3184.2051], skew=2.090, zero_frac=0.000, seasonal_mean_cv=0.199, reason=Outliers dominate more than skew; robust centering/scaling is safer than aggressive warping.
- psfc: recipe=identity -> no_extra_clip -> center=mean -> scale=std, clip=[4.894e+04, 1.019e+05], skew=-1.046, zero_frac=0.000, seasonal_mean_cv=0.003, reason=Moderate skew, but not severe enough to force a nonlinear transform.
- q2: recipe=identity -> no_extra_clip -> center=mean -> scale=std, clip=[0, 0.0247], skew=-0.147, zero_frac=0.001, seasonal_mean_cv=0.261, reason=Distribution is close enough to symmetric for standard scaling.
- rain: recipe=identity -> winsorize_to_p01_p99 -> center=median -> scale=iqr, clip=[0, 1.9324], skew=35.165, zero_frac=0.705, seasonal_mean_cv=1.168, reason=Outliers dominate more than skew; robust centering/scaling is safer than aggressive warping.
- swdown: recipe=identity -> no_extra_clip -> center=mean -> scale=std, clip=[0, 1159.567], skew=1.181, zero_frac=0.499, seasonal_mean_cv=0.214, reason=Moderate skew, but not severe enough to force a nonlinear transform.
- t2: recipe=identity -> no_extra_clip -> center=mean -> scale=std, clip=[238.1151, 319.5679], skew=-1.005, zero_frac=0.000, seasonal_mean_cv=0.014, reason=Moderate skew, but not severe enough to force a nonlinear transform.
- u10: recipe=identity -> no_extra_clip -> center=median -> scale=iqr, clip=[-13.6498, 18.9382], skew=0.279, zero_frac=0.000, seasonal_mean_cv=0.667, reason=Outliers dominate more than skew; robust centering/scaling is safer than aggressive warping.
- v10: recipe=identity -> no_extra_clip -> center=median -> scale=iqr, clip=[-16.2932, 22.2262], skew=0.246, zero_frac=0.000, seasonal_mean_cv=6.724, reason=Outliers dominate more than skew; robust centering/scaling is safer than aggressive warping.

## External References

- STL decomposition background: https://www.statsmodels.org/v0.12.2/examples/notebooks/generated/stl_decomposition.html
- Power transforms (Yeo-Johnson / Box-Cox): https://scikit-learn.org/1.3/modules/generated/sklearn.preprocessing.power_transform.html
- Robust scaling: https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.RobustScaler.html
- Quantile transform for extreme tail reshaping: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
