"""
EXP07 isolated training script: single-GPU + AMP + configurable sparse validation.

This file is intentionally separate from scripts/train.py so previous experiments are unaffected.

Usage:
    python scripts/train_exp07_singlegpu_amp.py --config configs/train.yaml
    python scripts/train_exp07_singlegpu_amp.py --config /kaggle/working/exp07_train.yaml --raw_path /path/to/raw
"""

import argparse
import json
import os
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import FNO2D, HybridFNOConvLSTMEnsemble, hybrid_total_loss
from src.data.raw_window_dataset import RawWindowDataset
from src.utils import competition_metrics as cmetrics
from src.utils.adam import Adam
from src.utils.config import load_config
from src.utils.preprocessing import get_feature_lists
from src.utils.utilities3 import LpLoss

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_num_threads(1)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _adapt_state_dict_for_model(state_dict, model):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if model_keys == ckpt_keys:
        return state_dict

    has_module = any(k.startswith("module.") for k in ckpt_keys)
    model_has_module = any(k.startswith("module.") for k in model_keys)

    if has_module and not model_has_module:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if (not has_module) and model_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--raw_path", type=str, default=None, help="Override cfg.paths.raw_path")
    return p.parse_args()


def count_params(model):
    return sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)


def alpha_for_epoch(epoch: int, main_start: int, fine_start: int, warmup: float, main: float, finetune: float) -> float:
    if epoch >= fine_start:
        return finetune
    if epoch >= main_start:
        return main
    return warmup


args = parse_args()
cfg = load_config(args.config)
raw_path = args.raw_path if args.raw_path is not None else cfg.paths.raw_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        raise RuntimeError(
            "CUDA runtime initialization failed. Restart kernel/session and rerun from top. "
            f"Original error: {exc}"
        ) from exc

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU: {gpu_name}")

torch.manual_seed(0)
np.random.seed(0)

# -----------------------
# Settings from YAML
# -----------------------

time_input = cfg.data.time_input
time_out = cfg.data.time_out
S1 = cfg.data.S1
S2 = cfg.data.S2

_, _, _, all_features, _ = get_feature_lists(cfg)
V = len(all_features)

batch_size = int(cfg.training.batch_size)
epochs = int(cfg.training.epochs)
validate_every = max(1, int(getattr(cfg.training, "validate_every", 1)))
validate_every_warmup_epochs = max(0, int(getattr(cfg.training, "validate_every_warmup_epochs", 0)))
validate_every_warmup = max(1, int(getattr(cfg.training, "validate_every_warmup", validate_every)))
use_amp = bool(getattr(cfg.training, "use_amp", True)) and device.type == "cuda"
max_grad_norm = float(getattr(cfg.training, "max_grad_norm", 1.0))


def validate_interval_for_epoch(epoch: int) -> int:
    if epoch < validate_every_warmup_epochs:
        return validate_every_warmup
    return validate_every

# =========================================================
# Dataloader (kept as-is optimized settings)
# =========================================================

train_dataset = RawWindowDataset(cfg, raw_path, "train")
test_dataset = RawWindowDataset(cfg, raw_path, "val")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

print(f"Train: {len(train_dataset)} samples | Val: {len(test_dataset)} samples")
print("DataParallel disabled in EXP07 script (single-GPU mode).")

# =========================================================
# Model
# =========================================================

model = FNO2D(
    time_in=time_input,
    features=V,
    time_out=time_out,
    width=cfg.model.width,
    modes=cfg.model.modes,
).to(device)

model_name = str(getattr(cfg.model, "name", "fno2d")).lower()
if model_name in ("hybrid", "ensemble", "hybrid_ensemble"):
    model = HybridFNOConvLSTMEnsemble(
        feature_names=all_features,
        time_in=time_input,
        time_out=time_out,
        width=int(getattr(cfg.model, "width", 64)),
        modes1=int(getattr(cfg.model, "modes1", getattr(cfg.model, "modes", 24))),
        modes2=int(getattr(cfg.model, "modes2", getattr(cfg.model, "modes", 24))),
        convlstm_hidden=int(getattr(cfg.model, "convlstm_hidden", 64)),
    ).to(device)

if model_name in ("hybrid", "ensemble", "hybrid_ensemble"):
    print(
        "HybridFNOConvLSTMEnsemble "
        f"| width={getattr(cfg.model, 'width', 64)} "
        f"| modes={getattr(cfg.model, 'modes1', getattr(cfg.model, 'modes', 24))}x{getattr(cfg.model, 'modes2', getattr(cfg.model, 'modes', 24))} "
        f"| clstm_hidden={getattr(cfg.model, 'convlstm_hidden', 64)} "
        f"| params={count_params(model):,}"
    )
else:
    print(f"FNO2D | width={cfg.model.width} | modes={cfg.model.modes} | params={count_params(model):,}")

optimizer = Adam(
    model.parameters(),
    lr=float(cfg.training.lr),
    weight_decay=float(cfg.training.weight_decay),
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.training.scheduler_step,
    gamma=cfg.training.scheduler_gamma,
)

scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
print(f"AMP enabled: {use_amp}")

myloss = LpLoss(size_average=False)

# =========================================================
# Checkpoint paths
# =========================================================

ckpt_dir = os.path.dirname(cfg.paths.model_save_path)
log_save = cfg.paths.save_dir

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_save), exist_ok=True)

best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
last_ckpt_path = os.path.join(ckpt_dir, "last.pt")

checkpoint_every = int(getattr(cfg.training, "checkpoint_every", 5))
resume_enabled = bool(getattr(cfg.training, "resume_from_best", True))
live_sync_enabled = bool(getattr(cfg.training, "live_sync_enabled", False))
live_sync_every = max(1, int(getattr(cfg.training, "live_sync_every", 1)))
live_sync_include_log = bool(getattr(cfg.training, "live_sync_include_log", True))

default_live_sync_dir = os.path.join(
    "/kaggle/working/live_checkpoints",
    os.path.basename(os.path.normpath(ckpt_dir)) if ckpt_dir else "run",
)
live_sync_dir = str(getattr(cfg.paths, "live_sync_dir", default_live_sync_dir))

log_competition_metrics = bool(getattr(cfg.training, "log_competition_metrics", True))
episode_sigma = float(getattr(cfg.training, "episode_sigma", 3.0))
episode_floor = float(getattr(cfg.training, "episode_floor", 1.0))
metric_w1 = float(getattr(cfg.training, "offline_metric_w1", 1.0 / 3.0))
metric_w2 = float(getattr(cfg.training, "offline_metric_w2", 1.0 / 3.0))
metric_w3 = float(getattr(cfg.training, "offline_metric_w3", 1.0 / 3.0))
checkpoint_on_offline_score = bool(getattr(cfg.training, "checkpoint_on_offline_score", False))
if checkpoint_on_offline_score and not log_competition_metrics:
    print("⚠️ checkpoint_on_offline_score disabled: set log_competition_metrics true to use it.")
    checkpoint_on_offline_score = False
best_surrogate = float("-inf")

episode_alpha_warmup = float(getattr(cfg.training, "episode_alpha_warmup", 1.0))
episode_alpha_main = float(getattr(cfg.training, "episode_alpha_main", 5.0))
episode_alpha_finetune = float(getattr(cfg.training, "episode_alpha_finetune", 8.0))
episode_main_start_epoch = int(getattr(cfg.training, "episode_main_start_epoch", 20))
episode_finetune_start_epoch = int(getattr(cfg.training, "episode_finetune_start_epoch", 50))
ep_head_weight = float(getattr(cfg.training, "ep_head_weight", 0.1))

# =========================================================
# Resume from checkpoint
# =========================================================

start_epoch = 0
best_val_loss = float("inf")
last_eval_val_loss = float("inf")
log = []

resume_path = None
if resume_enabled:
    if os.path.exists(last_ckpt_path):
        resume_path = last_ckpt_path
    elif os.path.exists(best_ckpt_path):
        resume_path = best_ckpt_path

if resume_path is not None:
    print(f"\n🔄 Resuming from {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    adapted_state_dict = _adapt_state_dict_for_model(ckpt["model_state_dict"], model)
    model.load_state_dict(adapted_state_dict)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_l2", ckpt.get("val_loss", float("inf")))
    last_eval_val_loss = ckpt.get("val_loss", best_val_loss)
    best_surrogate = ckpt.get("best_val_offline_surrogate", float("-inf"))

    if os.path.exists(log_save):
        with open(log_save, "r", encoding="utf-8") as f:
            log = json.load(f)

    print(f"  → Resuming from epoch {start_epoch}, best_val={best_val_loss:.6f}")
else:
    print("\n🆕 Starting fresh training (no checkpoint found)")


def save_checkpoint(path, epoch, val_l2):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_l2,
            "best_val_l2": best_val_loss,
            "best_val_offline_surrogate": best_surrogate,
        },
        path,
    )


def sync_live_artifacts(epoch: int, force: bool = False) -> None:
    if not live_sync_enabled:
        return
    if not force and ((epoch + 1) % live_sync_every != 0):
        return

    try:
        os.makedirs(live_sync_dir, exist_ok=True)
        copied = []

        if os.path.exists(best_ckpt_path):
            shutil.copy2(best_ckpt_path, os.path.join(live_sync_dir, "best.pt"))
            copied.append("best.pt")
        if os.path.exists(last_ckpt_path):
            shutil.copy2(last_ckpt_path, os.path.join(live_sync_dir, "last.pt"))
            copied.append("last.pt")
        if live_sync_include_log and os.path.exists(log_save):
            shutil.copy2(log_save, os.path.join(live_sync_dir, "log.json"))
            copied.append("log.json")

        if copied:
            print(f" 🔁 live sync → {live_sync_dir}", end="")
    except Exception as exc:
        print(f" ⚠️ live sync failed: {exc}", end="")


print(f"\nTraining epochs {start_epoch} → {epochs - 1} ({epochs - start_epoch} remaining)")
if validate_every_warmup_epochs > 0:
    print(
        "Validate schedule: "
        f"every {validate_every_warmup} epoch(s) for first {validate_every_warmup_epochs}, "
        f"then every {validate_every} epoch(s)"
    )
else:
    print(f"Validate every {validate_every} epoch(s)")
print(f"Checkpoint every {checkpoint_every} epochs → {ckpt_dir}")
if live_sync_enabled:
    print(f"Live sync enabled every {live_sync_every} epoch(s) → {live_sync_dir}")
print(f"{'=' * 60}\n")

for ep in tqdm(range(start_epoch, epochs), desc="Epoch", initial=start_epoch, total=epochs):
    t0 = time.time()
    cur_alpha = alpha_for_epoch(
        ep,
        episode_main_start_epoch,
        episode_finetune_start_epoch,
        episode_alpha_warmup,
        episode_alpha_main,
        episode_alpha_finetune,
    )

    model.train()
    train_l2 = 0.0

    for batch in train_loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, season_idx = batch
        else:
            x, y = batch
            season_idx = None

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if season_idx is not None:
            season_idx = season_idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if model_name in ("hybrid", "ensemble", "hybrid_ensemble"):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out, aux = model(x, season_idx=season_idx, return_aux=True)

            hist_cpm = x[..., 0]
            ep_mask = cmetrics.episode_proxy_mask(hist_cpm, y, sigma=episode_sigma, floor=episode_floor)
            with torch.amp.autocast("cuda", enabled=False):
                losses = hybrid_total_loss(
                    out.float(),
                    y.float(),
                    aux["ep_prob"].float(),
                    ep_mask,
                    alpha=cur_alpha,
                    ep_head_weight=ep_head_weight,
                )
            train_loss = losses["total"]
            train_metric_like = losses["main"]
        else:
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x).view(x.size(0), S1, S2, time_out)
                train_loss = myloss(out, y)
                train_metric_like = train_loss

        scaler.scale(train_loss).backward()
        scaler.unscale_(optimizer)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        train_l2 += train_metric_like.item()

    scheduler.step()

    cur_validate_every = validate_interval_for_epoch(ep)
    do_validate = ((ep + 1) % cur_validate_every == 0) or (ep == epochs - 1)
    test_l2 = None
    comp_sums = {
        k: 0.0
        for k in (
            "GlobalSMAPE",
            "EpisodeSMAPE",
            "EpisodeCorr",
            "NormGlobalSMAPE",
            "NormEpisodeSMAPE",
            "NormEpisodeCorr",
            "offline_surrogate",
        )
    }
    comp_batches = 0

    if do_validate:
        model.eval()
        eval_l2_sum = 0.0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, y, season_idx = batch
                else:
                    x, y = batch
                    season_idx = None

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if season_idx is not None:
                    season_idx = season_idx.to(device, non_blocking=True)

                if model_name in ("hybrid", "ensemble", "hybrid_ensemble"):
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out, aux = model(x, season_idx=season_idx, return_aux=True)

                    hist_cpm = x[..., 0]
                    ep_mask = cmetrics.episode_proxy_mask(hist_cpm, y, sigma=episode_sigma, floor=episode_floor)
                    with torch.amp.autocast("cuda", enabled=False):
                        losses = hybrid_total_loss(
                            out.float(),
                            y.float(),
                            aux["ep_prob"].float(),
                            ep_mask,
                            alpha=cur_alpha,
                            ep_head_weight=ep_head_weight,
                        )
                    eval_l2_sum += losses["main"].item()
                else:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = model(x).view(x.size(0), S1, S2, time_out)
                        eval_l2_sum += myloss(out, y).item()

                if log_competition_metrics:
                    hist_cpm = x[..., 0]
                    ep_mask = cmetrics.episode_proxy_mask(hist_cpm, y, sigma=episode_sigma, floor=episode_floor)
                    bundle = cmetrics.competition_metric_bundle(out, y, ep_mask, w1=metric_w1, w2=metric_w2, w3=metric_w3)
                    bs = x.size(0)
                    comp_batches += bs
                    for k in comp_sums:
                        comp_sums[k] += float(bundle[k].item()) * bs

        test_l2 = eval_l2_sum / len(test_dataset)
        last_eval_val_loss = test_l2

    train_l2 /= len(train_dataset)
    dur = time.time() - t0

    log_row = {
        "epoch": ep,
        "duration": dur,
        "train_l2": train_l2,
        "validated": bool(do_validate),
    }

    if do_validate and test_l2 is not None:
        log_row["val_l2"] = test_l2
        if log_competition_metrics and comp_batches > 0:
            for k, v in comp_sums.items():
                log_row[f"val_{k}"] = v / comp_batches
    else:
        log_row["val_l2"] = last_eval_val_loss

    log.append(log_row)

    if do_validate and test_l2 is not None:
        if log_competition_metrics and comp_batches > 0:
            gs = log_row.get("val_GlobalSMAPE", 0.0)
            es = log_row.get("val_EpisodeSMAPE", 0.0)
            ec = log_row.get("val_EpisodeCorr", 0.0)
            fs = log_row.get("val_offline_surrogate", 0.0)
            print(
                f"Ep {ep:3d} | {dur:.0f}s | train={train_l2:.5f} | val_l2={test_l2:.5f} | "
                f"val_GSMAPE={gs:.4f} val_ESMAPE={es:.4f} val_ECorr={ec:.4f} val_score≈{fs:.4f} alpha={cur_alpha:.2f}",
                end="",
            )
        else:
            print(f"Ep {ep:3d} | {dur:.0f}s | train={train_l2:.5f} | val={test_l2:.5f}", end="")
    else:
        print(f"Ep {ep:3d} | {dur:.0f}s | train={train_l2:.5f} | val=SKIP (every {cur_validate_every})", end="")

    try:
        import wandb

        if wandb.run is not None:
            wb = {
                "epoch": ep,
                "train_l2": train_l2,
                "val_l2": log_row["val_l2"],
                "duration_s": dur,
                "lr": optimizer.param_groups[0]["lr"],
                "validated": int(do_validate),
            }
            if do_validate and log_competition_metrics and comp_batches > 0:
                wb.update({k: log_row[k] for k in log_row if k.startswith("val_")})
            wandb.log(wb)
    except ImportError:
        pass

    if do_validate and test_l2 is not None:
        improved_l2 = test_l2 < best_val_loss
        if improved_l2:
            best_val_loss = test_l2

        val_surrogate = log_row.get("val_offline_surrogate")
        improved_surrogate = val_surrogate is not None and val_surrogate > best_surrogate
        if improved_surrogate:
            best_surrogate = val_surrogate

        if checkpoint_on_offline_score:
            save_best = improved_surrogate
        else:
            save_best = improved_l2

        if save_best:
            save_checkpoint(best_ckpt_path, ep, test_l2)
            sync_live_artifacts(ep)
            print(" ✅ BEST saved", end="")

    save_checkpoint(last_ckpt_path, ep, log_row["val_l2"])
    sync_live_artifacts(ep)

    if (ep + 1) % checkpoint_every == 0:
        with open(log_save, "w", encoding="utf-8") as f:
            json.dump(log, f)
        sync_live_artifacts(ep)
        print(" 📝 log saved", end="")

    print()

with open(log_save, "w", encoding="utf-8") as f:
    json.dump(log, f)
sync_live_artifacts(max(0, epochs - 1), force=True)

print(f"\n{'=' * 60}")
print("Training complete!")
print(f"  Best val_l2 = {best_val_loss:.6f}")
if log_competition_metrics and best_surrogate > float("-inf"):
    print(f"  Best val offline surrogate (higher better) = {best_surrogate:.6f}")
print(f"  Best checkpoint: {best_ckpt_path}")
print(f"  Last checkpoint: {last_ckpt_path}")
print(f"  Log: {log_save}")
print(f"{'=' * 60}")
