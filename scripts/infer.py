"""Inference from raw test ``.npy`` (same channel layout as training; targets in physical cpm25 units)."""

import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import FNO2D, HybridFNOConvLSTMEnsemble
from src.data.raw_window_dataset import base_keys_from_cfg, load_normalization_payload, normalize_raw_slice
from src.utils.config import load_config
from src.utils.preprocessing import build_window_features, get_feature_lists

warnings.filterwarnings("ignore")


def _adapt_state_dict_for_model(state_dict, model):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    model_state = model.state_dict()

    has_module = any(k.startswith("module.") for k in ckpt_keys)
    model_has_module = any(k.startswith("module.") for k in model_keys)

    if has_module and not model_has_module:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif (not has_module) and model_has_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    adapted = {}
    for k, v in state_dict.items():
        tgt = model_state.get(k)
        if tgt is None:
            adapted[k] = v
            continue

        # Spectral weights compatibility:
        # old: complex (Cin,Cout,M1,M2)
        # new: float (Cin,Cout,M1,M2,2) with real/imag in last dim
        if torch.is_complex(v) and (not torch.is_complex(tgt)) and v.ndim == 4 and tgt.ndim == 5 and tgt.shape[-1] == 2:
            adapted[k] = torch.view_as_real(v)
            continue
        if (not torch.is_complex(v)) and v.ndim == 5 and v.shape[-1] == 2 and torch.is_complex(tgt) and tgt.ndim == 4:
            adapted[k] = torch.view_as_complex(v.contiguous())
            continue
        if (not torch.is_complex(v)) and v.ndim == 5 and v.shape[0] == 2 and (not torch.is_complex(tgt)) and tgt.ndim == 5 and tgt.shape[-1] == 2:
            adapted[k] = torch.stack([v[0], v[1]], dim=-1)
            continue

        adapted[k] = v

    return adapted


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/infer.yaml")
    p.add_argument("--input_loc", type=str, default=None)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--output_loc", type=str, default=None)
    return p.parse_args()


class TestRawDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, input_loc: str):
        self.cfg = cfg
        self.input_loc = input_loc
        self.time_input = int(cfg.data.time_input)
        self.S1 = int(cfg.data.S1)
        self.S2 = int(cfg.data.S2)
        _, _, _, self.all_features, _ = get_feature_lists(cfg)
        self.base_keys = base_keys_from_cfg(cfg)
        pre_cfg = getattr(cfg, "preprocessing", None)
        stats_path = getattr(pre_cfg, "stats_path", None) if pre_cfg is not None else None
        self.norm_payload = load_normalization_payload(stats_path)
        self.arrs = {
            k: np.load(os.path.join(input_loc, f"{k}.npy"), mmap_mode="r") for k in self.base_keys
        }
        self.N = int(self.arrs[self.base_keys[0]].shape[0])
        self.V = len(self.all_features)

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        raw_slices = {
            k: np.asarray(self.arrs[k][idx, : self.time_input], dtype=np.float32) for k in self.base_keys
        }
        raw_slices = normalize_raw_slice(raw_slices, self.norm_payload)
        feats = build_window_features(raw_slices, self.all_features)
        x = np.empty((self.time_input, self.S1, self.S2, self.V), dtype=np.float32)
        for c, f in enumerate(self.all_features):
            x[..., c] = feats[f]
        return torch.from_numpy(x)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    input_loc = args.input_loc if args.input_loc is not None else cfg.paths.input_loc
    model_path = args.model_path if args.model_path is not None else cfg.paths.model_path
    output_loc = args.output_loc if args.output_loc is not None else cfg.paths.output_loc

    time_input = int(cfg.data.time_input)
    time_out = int(cfg.data.time_out)
    S1 = int(cfg.data.S1)
    S2 = int(cfg.data.S2)
    _, _, _, all_features, _ = get_feature_lists(cfg)
    V = len(all_features)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"INPUT_LOC: {input_loc}")

    if not os.path.isdir(input_loc):
        raise FileNotFoundError(f"input_loc is not a directory: {input_loc}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    test_dataset = TestRawDataset(cfg, input_loc)
    infer_cfg = getattr(cfg, "inference", None)
    infer_batch_size = int(getattr(infer_cfg, "batch_size", 1)) if infer_cfg is not None else 1

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    checkpoint = torch.load(model_path, map_location=device)
    print(f"Loaded model from: {model_path} (epoch {checkpoint.get('epoch', '?')})")

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
    else:
        model = FNO2D(
            time_in=time_input,
            features=V,
            time_out=time_out,
            width=cfg.model.width,
            modes=cfg.model.modes,
        ).to(device)

    mgpu_cfg = getattr(cfg, "multi_gpu", None)
    mgpu_enabled = bool(getattr(mgpu_cfg, "enabled", False)) if mgpu_cfg is not None else False
    mgpu_backend = str(getattr(mgpu_cfg, "backend", "data_parallel")).lower() if mgpu_cfg is not None else "data_parallel"
    mgpu_min_gpus = int(getattr(mgpu_cfg, "min_gpus", 2)) if mgpu_cfg is not None else 2
    mgpu_device_ids = getattr(mgpu_cfg, "device_ids", None) if mgpu_cfg is not None else None

    if device.type == "cuda" and mgpu_enabled and mgpu_backend == "data_parallel":
        gpu_count = torch.cuda.device_count()
        if gpu_count >= mgpu_min_gpus:
            if isinstance(mgpu_device_ids, list) and len(mgpu_device_ids) > 0:
                model = nn.DataParallel(model, device_ids=[int(x) for x in mgpu_device_ids])
            else:
                model = nn.DataParallel(model)
            active_ids = model.device_ids if isinstance(model, nn.DataParallel) else list(range(gpu_count))
            print(f"✅ DataParallel enabled for inference on GPUs: {active_ids}")
        else:
            print(f"ℹ️ multi_gpu.enabled=True but only {gpu_count} GPU(s) available (< min_gpus={mgpu_min_gpus}); running single-GPU inference.")

    adapted_state_dict = _adapt_state_dict_for_model(checkpoint["model_state_dict"], model)
    model.load_state_dict(adapted_state_dict)
    model.eval()

    out_dir = output_loc if os.path.isdir(output_loc) else os.path.dirname(output_loc)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    preds_path = os.path.join(output_loc, "preds.npy") if os.path.isdir(output_loc) else output_loc

    prediction = np.zeros((len(test_dataset), S1, S2, time_out), dtype=np.float32)

    cursor = 0
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(device, non_blocking=True)
            if model_name in ("hybrid", "ensemble", "hybrid_ensemble"):
                infer_season_idx = getattr(cfg.model, "infer_season_idx", None)
                season_idx_tensor = None
                if infer_season_idx is not None:
                    season_idx_tensor = torch.full(
                        (x.shape[0],),
                        int(infer_season_idx),
                        device=device,
                        dtype=torch.long,
                    )
                out = model(x, season_idx=season_idx_tensor)
            else:
                out = model(x)

            if out.ndim != 4:
                raise ValueError(f"Expected 4D model output, got shape {tuple(out.shape)}")
            if out.shape[1] == time_out and out.shape[-1] != time_out:
                out = out.permute(0, 2, 3, 1)
            if out.shape[1:] != (S1, S2, time_out):
                raise ValueError(
                    f"Unexpected inference output shape {tuple(out.shape)}; "
                    f"expected (B, {S1}, {S2}, {time_out})"
                )

            out_np = out.cpu().numpy()
            bs = out_np.shape[0]
            prediction[cursor: cursor + bs] = out_np
            cursor += bs

    prediction = np.clip(prediction, 0.0, None)
    np.save(preds_path, prediction)
    print(f"Saved predictions to: {preds_path}")


if __name__ == "__main__":
    main()
