# plan-dualGpuTrainingSpeed

## Highest impact (do now)

- Add AMP in `scripts/train.py`:
  - Use `torch.amp.autocast("cuda")` and `GradScaler`.
  - Keep gradient clipping after `scaler.unscale_(optimizer)`.
- Keep 2-GPU `DataParallel` enabled.
- Keep global `BATCH_SIZE=8` for 2x T4 (4 per GPU).
- Reduce epochs from `65` to `28` for deadline-safe turnaround.
- Validate every `3` epochs instead of every epoch.

## Already good (keep as-is)

- DataLoader settings in training are already optimized:
  - `num_workers=4`
  - `pin_memory=True`
  - `persistent_workers=True`
  - `prefetch_factor` enabled
- Episode mask computation is vectorized (`episode_proxy_mask`) and not STL-per-sample.

## Runtime target

- Goal: reduce from ~25 min/epoch to ~6–12 min/epoch.
- With 28 epochs, target total training runtime: ~3–6 hours.

## Execution order

1. Enable AMP in training script.
2. Add `validate_every=3` logic.
3. Set notebook constants:
   - `EPOCHS=28`
   - `TRAIN_BATCH_SIZE_2GPU=8`
4. Run full pipeline from top cells after kernel restart.
5. Check first epoch timing; if still high, apply model-size reduction.
