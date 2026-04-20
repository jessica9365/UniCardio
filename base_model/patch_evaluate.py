with open('evaluate_bp.py', 'r') as f:
    content = f.read()

with open('evaluate_bp.py.bak', 'w') as f:
    f.write(content)

# Swap file paths
content = content.replace(
    "torch.load('input_batch_mimic0.pt',   weights_only=False)",
    "torch.load('input_batch_vitaldb.pt',  weights_only=False)"
)
content = content.replace(
    "torch.load('results_batch_mimic0.pt', weights_only=False)",
    "torch.load('results_batch_vitaldb.pt', weights_only=False)"
)

# GT is already (N, 1, 500) — no avg_pool needed
content = content.replace(
    "gt_ds    = F.avg_pool1d(gt, kernel_size=4, stride=4)  # 2000→500 → [375, 1, 500]",
    "gt_ds    = gt  # [VitalDB] already (N, 1, 500) — no pooling needed"
)

with open('evaluate_bp.py', 'w') as f:
    f.write(content)

print("✅ evaluate_bp.py patched")
