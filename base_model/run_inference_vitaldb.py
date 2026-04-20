import torch
import numpy as np
import yaml
from diffusion_model_no_compress_final import diff_CSDI, CSDI_base

device = torch.device("cuda")

with open('base_no_compress_original.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ── Load VitalDB test data ──
data = np.load('/cpfs01/shane/jess_data/data/archive/unicardio_vitaldb_test.npz')
ppg = torch.tensor(data['ppg'], dtype=torch.float32)  # (N, 500) → ch0: 0:500
ecg = torch.tensor(data['ecg'], dtype=torch.float32)  # (N, 500) → ch1: 500:1000
abp = torch.tensor(data['abp'], dtype=torch.float32)  # (N, 500) → ch3: 1500:2000
N   = ppg.shape[0]

# Layout: [PPG(0:500) | ECG(500:1000) | zeros(1000:1500) | ABP(1500:2000)]
dummy   = torch.zeros(N, 500)
flat    = torch.cat([ppg, ecg, dummy, abp], dim=1)       # (N, 2000)
signal  = flat.unsqueeze(1)                               # (N, 1, 2000)
print(f"Input shape: {signal.shape}")

# ── Load pretrained model ──
Model = CSDI_base(config, device, L=500*4).to(device)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load("no_compress799.pth", weights_only=False))
Model.eval()
print("Model loaded ✅")

# ── Inference: PPG(ch0) → ABP(ch3), model_flag='013' ──
BATCH_SIZE = 64
all_results = []
all_inputs  = []

for i in range(0, N, BATCH_SIZE):
    batch_sig = signal[i:i+BATCH_SIZE].to(device)   # (B, 1, 2000)

    with torch.no_grad():
        results = Model(
            batch_sig,
            n_samples=10,
            model_flag='013',    # PPG(ch0) → ABP(ch3)
            borrow_mode=2,
            DDIM_flag=1,
            sample_steps=6,
            train_gen_flag=1
        )

    all_results.append(results[0].cpu())
    all_inputs.append(batch_sig.cpu())

    if i % 320 == 0:
        print(f"  {i}/{N} done...")

all_results = torch.cat(all_results, dim=0)  # (N, 50, 1, 500)
all_inputs  = torch.cat(all_inputs,  dim=0)  # (N, 1, 2000)

# Ground truth ABP is at positions 1500:2000
gt_abp = flat[:, 1500:2000].unsqueeze(1)     # (N, 1, 500)
torch.save(gt_abp,      'input_batch_vitaldb.pt')
torch.save(all_results, 'results_batch_vitaldb.pt')
print(f"✅ Done — results: {all_results.shape}")
print("Run: python3 evaluate_bp.py")
