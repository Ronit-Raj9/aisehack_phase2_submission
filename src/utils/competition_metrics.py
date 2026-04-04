"""
Competition PM2.5 metrics (Phase 2) — aligned with official definitions.

Notation
--------
- y_{i,t}: ground-truth PM2.5 at grid i, forecast timestep t
- ŷ_{i,t}: prediction at (i, t)
- T = 16: forecast horizon
- E_t: episode grid indices at time t (boolean mask over space)
- G: full spatial domain (140×124)

GlobalSMAPE
-----------
For each t:  SMAPE_t^global = (1/|G|) Σ_{i∈G} |ŷ-y| / (0.5(|y|+|ŷ|))
Then:       GlobalSMAPE = (1/T) Σ_t SMAPE_t^global
Batch: mean over samples.

EpisodeSMAPE
------------
For each t: SMAPE_t^episode = (1/|E_t|) Σ_{i∈E_t} |pred-target| / (0.5(|y|+|pred|)) when |E_t|>0.
If |E_t|=0, that timestep contributes **0** (competition formulas assume defined per-t terms).
EpisodeSMAPE = (1/|T|) Σ_{t∈T} SMAPE_t^episode  with |T|=16.

EpisodeCorr
-----------
For each t: Corr_t = Pearson(y, ŷ) over i ∈ E_t when |E_t|≥2; else Corr_t = **0**.
EpisodeCorr = (1/|T|) Σ_{t∈T} Corr_t.

Normalization (higher is better for each)
-----------------------------------------
  NormGlobalSMAPE   = 1 - GlobalSMAPE / 2
  NormEpisodeSMAPE  = 1 - EpisodeSMAPE / 2
  NormEpisodeCorr   = (EpisodeCorr + 1) / 2

Final score (platform weights w1,w2,w3 undisclosed; unknown to submitters)
---------------------------------------------------------------------------
  Score = w1·NormGlobalSMAPE + w2·NormEpisodeCorr + w3·NormEpisodeSMAPE

Tensor layout: pred and target (B, H, W, T). episode_mask: bool (B, H, W, T).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

try:
    import numpy as np

    _HAS_NP = True
except ImportError:
    _HAS_NP = False


def episode_proxy_mask(
    history_raw: torch.Tensor,
    target_raw: torch.Tensor,
    sigma: float = 2.0,
    floor: float = 1.0,
) -> torch.Tensor:
    """
    Proxy episode mask when official E_t labels are unavailable.
    history_raw: (B, T_in, H, W) — past PM2.5 (physical scale)
    target_raw:  (B, H, W, T_out) — future PM2.5
    """
    baseline = history_raw.mean(dim=1, keepdim=False).unsqueeze(-1)
    spread = history_raw.std(dim=1, unbiased=False, keepdim=False).unsqueeze(-1)
    return (target_raw > baseline + sigma * spread) & (target_raw > floor)


def _as_bool_mask(episode_mask: torch.Tensor) -> torch.Tensor:
    if episode_mask.dtype != torch.bool:
        return episode_mask > 0.5
    return episode_mask


def _symmetric_smape_terms(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    num = (pred - target).abs()
    denom = 0.5 * (pred.abs() + target.abs())
    denom = denom.clamp_min(eps)
    return num / denom


def global_smape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scalar mean over batch: each sample uses competition GlobalSMAPE, then average B.
    """
    sm = _symmetric_smape_terms(pred, target, eps)
    per_t = sm.mean(dim=(1, 2))
    per_sample = per_t.mean(dim=1)
    return per_sample.mean()


def global_smape_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B,) GlobalSMAPE per batch element."""
    sm = _symmetric_smape_terms(pred, target, eps)
    return sm.mean(dim=(1, 2, 3))


def episode_smape(
    pred: torch.Tensor,
    target: torch.Tensor,
    episode_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Batch mean of per-sample EpisodeSMAPE = (1/|T|) Σ_t SMAPE_t^episode."""
    per_sample = episode_smape_per_sample(pred, target, episode_mask, eps=eps)
    return per_sample.mean()


def episode_smape_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    episode_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """(B,) EpisodeSMAPE with exact (1/T) timestep average; empty E_t contributes 0."""
    m = _as_bool_mask(episode_mask).float()
    sm = _symmetric_smape_terms(pred, target, eps)
    num_o = (sm * m).sum(dim=(1, 2))
    den_o = m.sum(dim=(1, 2))
    valid = den_o > 0
    smape_t = torch.where(valid, num_o / den_o.clamp_min(eps), torch.zeros_like(num_o))
    return smape_t.mean(dim=1)


def _pearson_1d(p: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    if p.numel() < 2:
        return p.new_tensor(0.0)
    p = p - p.mean()
    y = y - y.mean()
    denom = torch.sqrt((p * p).sum() * (y * y).sum()).clamp_min(eps)
    return (p * y).sum() / denom


def episode_corr(
    pred: torch.Tensor,
    target: torch.Tensor,
    episode_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Batch mean of per-sample EpisodeCorr = (1/|T|) Σ_t Corr_t."""
    return episode_corr_per_sample(pred, target, episode_mask, eps=eps).mean()


def episode_corr_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    episode_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """(B,) EpisodeCorr with (1/T) average; |E_t|<2 ⇒ Corr_t = 0."""
    episode_mask = _as_bool_mask(episode_mask)
    B, _, _, Tsteps = pred.shape
    out = []
    for b in range(B):
        corrs_t = []
        for t in range(Tsteps):
            m = episode_mask[b, :, :, t]
            pb = pred[b, :, :, t][m]
            yb = target[b, :, :, t][m]
            if pb.numel() >= 2:
                corrs_t.append(_pearson_1d(pb, yb, eps))
            else:
                corrs_t.append(pred.new_tensor(0.0))
        out.append(torch.stack(corrs_t).mean())
    return torch.stack(out)


def normalize_metrics(
    global_smape_val: torch.Tensor,
    episode_smape_val: torch.Tensor,
    episode_corr_val: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    norm_gs = 1.0 - global_smape_val / 2.0
    norm_es = 1.0 - episode_smape_val / 2.0
    norm_ec = (episode_corr_val + 1.0) / 2.0
    return norm_gs, norm_es, norm_ec


def competition_final_score(
    norm_global_smape: torch.Tensor,
    norm_episode_smape: torch.Tensor,
    norm_episode_corr: torch.Tensor,
    w1: float,
    w2: float,
    w3: float,
) -> torch.Tensor:
    """Weighted normalized score; higher is better (same as platform formula structure)."""
    return w1 * norm_global_smape + w2 * norm_episode_corr + w3 * norm_episode_smape


def offline_surrogate(
    norm_global_smape: torch.Tensor,
    norm_episode_smape: torch.Tensor,
    norm_episode_corr: torch.Tensor,
    w1: float,
    w2: float,
    w3: float,
) -> torch.Tensor:
    """Alias for local training when true w1,w2,w3 are unknown."""
    return competition_final_score(norm_global_smape, norm_episode_smape, norm_episode_corr, w1, w2, w3)


def competition_metric_bundle(
    pred: torch.Tensor,
    target: torch.Tensor,
    episode_mask: torch.Tensor,
    w1: float = 1.0 / 3.0,
    w2: float = 1.0 / 3.0,
    w3: float = 1.0 / 3.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Raw metrics, normalized components, and surrogate final score."""
    gs = global_smape(pred, target, eps=eps)
    es = episode_smape(pred, target, episode_mask, eps=eps)
    ec = episode_corr(pred, target, episode_mask, eps=eps)
    ng, ne, nc = normalize_metrics(gs, es, ec)
    score = competition_final_score(ng, ne, nc, w1=w1, w2=w2, w3=w3)
    return {
        "GlobalSMAPE": gs,
        "EpisodeSMAPE": es,
        "EpisodeCorr": ec,
        "NormGlobalSMAPE": ng,
        "NormEpisodeSMAPE": ne,
        "NormEpisodeCorr": nc,
        # w1·NormG + w2·NormEC + w3·NormES (true platform weights unknown)
        "offline_surrogate": score,
    }


def episode_corr_loss(pred: torch.Tensor, target: torch.Tensor, episode_mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - episode_corr(pred, target, episode_mask)


# --- NumPy (evaluation / offline analysis; same definitions) ---


def _np_smape_terms(pred, target, eps: float):
    num = np.abs(pred - target)
    denom = np.maximum(0.5 * (np.abs(pred) + np.abs(target)), eps)
    return num / denom


def global_smape_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    if not _HAS_NP:
        raise ImportError("numpy required for global_smape_np")
    sm = _np_smape_terms(pred, target, eps)
    per_t = sm.mean(axis=(1, 2))
    per_sample = per_t.mean(axis=1)
    return float(per_sample.mean())


def episode_smape_np(
    pred: np.ndarray,
    target: np.ndarray,
    episode_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    if not _HAS_NP:
        raise ImportError("numpy required for episode_smape_np")
    m = episode_mask.astype(np.float64)
    sm = _np_smape_terms(pred, target, eps)
    num_o = (sm * m).sum(axis=(1, 2))
    den_o = m.sum(axis=(1, 2))
    valid = den_o > 0
    smape_t = np.where(valid, num_o / np.maximum(den_o, eps), 0.0)
    per_sample = smape_t.mean(axis=1)
    return float(per_sample.mean())


def episode_corr_np(
    pred: np.ndarray,
    target: np.ndarray,
    episode_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    if not _HAS_NP:
        raise ImportError("numpy required for episode_corr_np")
    episode_mask = episode_mask.astype(bool)
    B, _, _, T = pred.shape
    per_sample = []
    for b in range(B):
        corrs_t = []
        for t in range(T):
            m = episode_mask[b, :, :, t]
            pb = pred[b, :, :, t][m]
            yb = target[b, :, :, t][m]
            if pb.size >= 2:
                pb = pb - pb.mean()
                yb = yb - yb.mean()
                denom = float(np.sqrt(np.sum(pb * pb) * np.sum(yb * yb)))
                corrs_t.append(float(np.sum(pb * yb) / denom) if denom >= eps else 0.0)
            else:
                corrs_t.append(0.0)
        per_sample.append(float(np.mean(corrs_t)))
    return float(np.mean(per_sample))


def competition_metric_bundle_np(
    pred: np.ndarray,
    target: np.ndarray,
    episode_mask: np.ndarray,
    w1: float = 1.0 / 3.0,
    w2: float = 1.0 / 3.0,
    w3: float = 1.0 / 3.0,
    eps: float = 1e-8,
) -> Dict[str, float]:
    gs = global_smape_np(pred, target, eps=eps)
    es = episode_smape_np(pred, target, episode_mask, eps=eps)
    ec = episode_corr_np(pred, target, episode_mask, eps=eps)
    ng = 1.0 - gs / 2.0
    ne = 1.0 - es / 2.0
    nc = (ec + 1.0) / 2.0
    s = w1 * ng + w2 * nc + w3 * ne
    return {
        "GlobalSMAPE": gs,
        "EpisodeSMAPE": es,
        "EpisodeCorr": ec,
        "NormGlobalSMAPE": ng,
        "NormEpisodeSMAPE": ne,
        "NormEpisodeCorr": nc,
        "offline_surrogate": s,
        "FinalScore": s,
    }
