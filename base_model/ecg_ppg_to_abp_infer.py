import torch
import numpy as np
import yaml
from diffusion_model_no_compress_final import diff_CSDI, CSDI_base

device = torch.device("cuda")

with open('base_no_compress_original.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 1) Load VitalDB test split
data  = np.load('/cpfs01/shane/jess_data/data/archive/unicardio_vitaldb_test.npz')
ppg   = torch.tensor(data['ppg'], dtype=torch.float32)  # (N, 500)
ecg   = torch.tensor(data['ecg'], dtype=torch.float32)  # (N, 500)
abp   = torch.tensor(data['abp'], dtype=torch.float32)  # (N, 500)
N     = ppg.shape[0]
dummy = torch.zeros(N, 500)

# 2) Pack into 4-channel flattened layout: ch0=PPG, ch1=ECG, ch2=dummy, ch3=ABP
flat   = torch.cat([ppg, ecg, dummy, abp], dim=1)   # (N, 2000)
signal = flat.unsqueeze(1)                          # (N, 1, 2000)
print("Input shape:", signal.shape)

# 3) Load pretrained UniCardio
Model = CSDI_base(config, device, L=500*4).to(device)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load("no_compress799.pth", weights_only=False))
Model.eval()
print("Model loaded ✅")

# 4) Run inference: ECG+PPG -> ABP
# NOTE: two-condition flags are len=3, so we must use generate()
from diffusion_model_no_compress_final import CSDI_base as _C
BATCH_SIZE = 64
all_pred   = []

for i in range(0, N, BATCH_SIZE):
    batch_sig = signal[i:i+BATCH_SIZE].to(device)  # (B, 1, 2000)
    with torch.no_grad():
        # generate() expects (B, K, L) with K=1 here; model_flag '013' = cond ch0+ch1, target ch3
        samples = Model.module.generate(
            batch_sig, 
            n_samples=3,
            model_flag='013',   # PPG (0) + ECG (1) -> ABP (3)
            borrow_mode=2,
            sample_steps=6,
            DDIM_flag=0,
            ratio=1,
            improved=0
        )                      # (B, 3, 1, 500)
    all_pred.append(samples.cpu())
    print(f"{min(i+BATCH_SIZE, N)}/{N} done")

pred = torch.cat(all_pred, dim=0)  # (N, 3, 1, 500)
gt   = flat[:, 1500:2000].unsqueeze(1)  # (N, 1, 500)

print("Pred shape:", pred.shape, "GT shape:", gt.shape)

torch.save(gt,   'input_batch_vitaldb.pt')
torch.save(pred, 'results_batch_vitaldb.pt')
print("✅ Saved input_batch_vitaldb.pt and results_batch_vitaldb.pt")
