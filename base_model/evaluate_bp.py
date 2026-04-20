import torch
import numpy as np
from scipy import stats
import torch.nn.functional as F

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def min_rmse(pred, target, max_shift=10):
    best = float('inf')
    for shift in range(-max_shift, max_shift + 1):
        shifted = torch.roll(target, shift, dims=-1)
        best = min(best, rmse(pred, shifted))
    return best

def min_mae(pred, target, max_shift=10):
    best = float('inf')
    for shift in range(-max_shift, max_shift + 1):
        shifted = torch.roll(target, shift, dims=-1)
        best = min(best, mae(pred, shifted))
    return best

def ks_test(pred, target):
    stat, p = stats.ks_2samp(
        pred.numpy().flatten(),
        target.numpy().flatten()
    )
    return stat, p

# ── Load ──────────────────────────────────────────────────────────────────────
gt  = torch.load('input_batch_vitaldb.pt',  weights_only=False).float().cpu()
gen = torch.load('results_batch_vitaldb.pt', weights_only=False).float().cpu()

# ── Align shapes ──────────────────────────────────────────────────────────────
gen_mean = gen.mean(dim=1)           # avg 50 diffusion samples → [375, 1, 500]
gt_ds    = gt  # [VitalDB] already (N, 1, 500) — no pooling needed

gen_flat = gen_mean[:, 0, :]  # generated ABP         # [375, 500]
gt_flat  = gt_ds[:, 0, :]   # [VitalDB] only channel is ABP

# ── Print raw stats ───────────────────────────────────────────────────────────
print("Before rescaling:")
print(f"  GT  — mean: {gt_flat.mean():.3f}, std: {gt_flat.std():.3f}")
print(f"  Gen — mean: {gen_flat.mean():.3f}, std: {gen_flat.std():.3f}")

# ── Rescale generated to match GT distribution ────────────────────────────────
# z-score normalize gen, then rescale to GT's mean/std
gt_mean  = gt_flat.mean()
gt_std   = gt_flat.std()
gen_mean_val = gen_flat.mean()
gen_std_val  = gen_flat.std()

gen_rescaled = (gen_flat - gen_mean_val) / (gen_std_val + 1e-8) * gt_std + gt_mean

print("\nAfter rescaling gen to match GT distribution:")
print(f"  GT          — mean: {gt_flat.mean():.3f}, std: {gt_flat.std():.3f}")
print(f"  Gen rescaled — mean: {gen_rescaled.mean():.3f}, std: {gen_rescaled.std():.3f}")

# ── Evaluate both raw and rescaled ────────────────────────────────────────────
for label, gen_eval in [("Raw (no rescaling)", gen_flat), ("Rescaled", gen_rescaled)]:
    r     = rmse(gen_eval, gt_flat)
    m     = mae(gen_eval, gt_flat)
    mr    = min_rmse(gen_eval, gt_flat, max_shift=10)
    mm    = min_mae(gen_eval, gt_flat, max_shift=10)
    ks, p = ks_test(gen_eval, gt_flat)

    print(f"\n{'=' * 55}")
    print(f"  [{label}] BP Generation — MIMIC-II")
    print(f"{'=' * 55}")
    print(f"  RMSE         : {r:.4f}")
    print(f"  Min-RMSE     : {mr:.4f}")
    print(f"  MAE          : {m:.4f}")
    print(f"  Min-MAE      : {mm:.4f}")
    print(f"  KS Statistic : {ks:.4f}  (p={p:.4e})")

# ── Best-of-50 with rescaling ─────────────────────────────────────────────────
print(f"\n--- Best-of-50 oracle (rescaled) ---")
best_rmse_list = []
for i in range(gt_flat.shape[0]):
    gt_i = gt_flat[i]                    # [500]
    cands = gen[i, :, 0, :].cpu()        # [50, 500]
    # rescale each candidate
    cands_r = (cands - cands.mean()) / (cands.std() + 1e-8) * gt_std + gt_mean
    rmses = torch.sqrt(((cands_r - gt_i.unsqueeze(0)) ** 2).mean(dim=-1))
    best_rmse_list.append(rmses.min().item())
print(f"  Best-of-50 RMSE: {np.mean(best_rmse_list):.4f}")

# ── mmHg conversion ──────────────────────────────────────────────────────────
ABP_MEAN = 82.26
ABP_STD  = 22.55
print("\n[mmHg Results]")
print(f"  RMSE : {r * ABP_STD:.2f} mmHg")
print(f"  MAE  : {m * ABP_STD:.2f} mmHg")
print(f"  Paper target (UniCardio-F): 5.79 mmHg RMSE")
