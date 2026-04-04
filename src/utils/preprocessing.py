from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-8

DERIVED_FEATURE_DEPENDENCIES = {
    "wind_speed": ["u10", "v10"],
    "wind_dir_cos": ["u10", "v10"],
    "wind_dir_sin": ["u10", "v10"],
    "rain_occ": ["rain"],
    "rain_amt": ["rain"],
    "daylight_mask": ["swdown"],
    "cpm25_delta1": ["cpm25"],
    "cpm25_delta3": ["cpm25"],
    "cpm25_mean3": ["cpm25"],
    "cpm25_mean6": ["cpm25"],
    "cpm25_std10": ["cpm25"],
    "pblh_inverse": ["pblh"],
    "ventilation_proxy": ["u10", "v10", "pblh"],
}


def get_feature_lists(
    cfg,
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Return model met channels, emissions, derived, full channel order, and physics-only met."""
    raw_met = list(
        getattr(cfg.features, "met_variables_raw", getattr(cfg.features, "met_variables", []))
    )
    persist_physics = list(getattr(cfg.features, "persist_met_for_physics", []))
    raw_emission = list(
        getattr(cfg.features, "emission_variables_raw", getattr(cfg.features, "emission_variables", []))
    )
    derived = list(getattr(cfg.features, "derived_variables", []))
    disk_met: List[str] = list(raw_met)
    for m in persist_physics:
        if m not in disk_met:
            disk_met.append(m)
    all_features = disk_met + raw_emission + derived
    return raw_met, raw_emission, derived, all_features, persist_physics


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    csum = np.cumsum(arr.astype(np.float64), axis=0)
    out = csum.copy()
    out[window:] = out[window:] - csum[:-window]
    return out


def causal_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    arr = arr.astype(np.float64)
    sums = _rolling_sum(arr, window)
    counts = np.minimum(np.arange(1, arr.shape[0] + 1, dtype=np.float64), float(window))
    return (sums / counts[:, None, None]).astype(np.float32)


def causal_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    arr = arr.astype(np.float64)
    sums = _rolling_sum(arr, window)
    sums_sq = _rolling_sum(arr * arr, window)
    counts = np.minimum(np.arange(1, arr.shape[0] + 1, dtype=np.float64), float(window))
    mean = sums / counts[:, None, None]
    mean_sq = sums_sq / counts[:, None, None]
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32)


def lag_delta(arr: np.ndarray, lag: int) -> np.ndarray:
    arr = arr.astype(np.float32)
    delta = np.zeros_like(arr, dtype=np.float32)
    delta[lag:] = arr[lag:] - arr[:-lag]
    return delta


def required_raw_features(feature: str) -> List[str]:
    return DERIVED_FEATURE_DEPENDENCIES.get(feature, [feature])


def compute_feature_array(feature: str, arrays: Dict[str, np.ndarray]) -> np.ndarray:
    if feature not in DERIVED_FEATURE_DEPENDENCIES:
        return arrays[feature].astype(np.float32)

    if feature == "wind_speed":
        u10 = arrays["u10"].astype(np.float32)
        v10 = arrays["v10"].astype(np.float32)
        return np.sqrt(np.maximum(u10 * u10 + v10 * v10, 0.0)).astype(np.float32)

    if feature == "wind_dir_cos":
        u10 = arrays["u10"].astype(np.float32)
        v10 = arrays["v10"].astype(np.float32)
        speed = np.sqrt(np.maximum(u10 * u10 + v10 * v10, 0.0))
        return (u10 / np.maximum(speed, EPS)).astype(np.float32)

    if feature == "wind_dir_sin":
        u10 = arrays["u10"].astype(np.float32)
        v10 = arrays["v10"].astype(np.float32)
        speed = np.sqrt(np.maximum(u10 * u10 + v10 * v10, 0.0))
        return (v10 / np.maximum(speed, EPS)).astype(np.float32)

    if feature == "rain_occ":
        return (arrays["rain"] > 0).astype(np.float32)

    if feature == "rain_amt":
        return np.maximum(arrays["rain"].astype(np.float32), 0.0)

    if feature == "daylight_mask":
        return (arrays["swdown"] > 0).astype(np.float32)

    if feature == "cpm25_delta1":
        return lag_delta(arrays["cpm25"], 1)

    if feature == "cpm25_delta3":
        return lag_delta(arrays["cpm25"], 3)

    if feature == "cpm25_mean3":
        return causal_rolling_mean(arrays["cpm25"], 3)

    if feature == "cpm25_mean6":
        return causal_rolling_mean(arrays["cpm25"], 6)

    if feature == "cpm25_std10":
        return causal_rolling_std(arrays["cpm25"], 10)

    if feature == "pblh_inverse":
        return (1.0 / np.maximum(arrays["pblh"].astype(np.float32), EPS)).astype(np.float32)

    if feature == "ventilation_proxy":
        u10 = arrays["u10"].astype(np.float32)
        v10 = arrays["v10"].astype(np.float32)
        pblh = arrays["pblh"].astype(np.float32)
        speed = np.sqrt(np.maximum(u10 * u10 + v10 * v10, 0.0))
        return (speed * pblh).astype(np.float32)

    raise KeyError(f"Unknown derived feature: {feature}")


def build_window_features(
    raw_slice: Dict[str, np.ndarray],
    output_feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """Derive channels using only ``raw_slice`` time context; raw values, float32 (no scaling)."""
    out: Dict[str, np.ndarray] = {}
    for feat in output_feature_names:
        raw = raw_slice.get(feat)
        if raw is None:
            raw = compute_feature_array(feat, raw_slice)
        out[feat] = raw.astype(np.float32)
    return out
