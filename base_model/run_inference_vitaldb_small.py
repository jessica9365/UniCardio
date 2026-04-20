import torch
import numpy as np
import yaml
from diffusion_model_no_compress_final import diff_CSDI, CSDI_base

device = torch.device("cuda")
with open('base_no_compress_original.yaml', 'r') as f:
    config = yaml.safe_load(f)

data  = np.load('/cpfs01/shane/jess_data/data/archive/unicardio_vitaldb_test.npz')
ppg   = torch.tensor(data['ppg'][:256], dtype=torch.float32)
ecg   = torch.tensor(data['ecg'][:256], dtype=torch.float32)
abp   = torch.tensor(data['abp'][:256], dtype=torch.float32)
N     = ppg.shape[0]
dummy = torch.zeros(N, 500)
flat  = torch.cat([ppg, ecg, dummy, abp], dim=1)
signal = flat.unsqueeze(1)
print(f"Input shape: {signal.shape}")

Model = CSDI_base(config, device, L=500*4).to(device)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load("no_compress799.pth", weights_only=False))
Model.eval()
print("Model loaded ✅")

BATCH_SIZE  = 64
all_results = []

for i in range(0, N, BATCH_SIZE):
    batch_sig = signal[i:i+BATCH_SIZE].to(device)
    with torch.no_grad():
        results = Model(batch_sig, n_samples=3, model_flag='013',
                        borrow_mode=2, DDIM_flag=0,
                        sample_steps=6, train_gen_flag=1)
    # results[0] shape: (B, n_samples, 1, 500) — keep all dims
    print(f"  batch results[0] shape: {results[0].shape}")
    all_results.append(results[0].cpu())
    print(f"  {i+BATCH_SIZE}/{N} done...")

all_results = torch.cat(all_results, dim=0)  # (256, n_samples, 1, 500)
gt_abp      = flat[:, 1500:2000].unsqueeze(1) # (256, 1, 500)

print(f"Final results shape: {all_results.shape}")
print(f"GT shape:            {gt_abp.shape}")

torch.save(gt_abp,      'input_batch_vitaldb.pt')
torch.save(all_results, 'results_batch_vitaldb.pt')
print("✅ Done — run: python3 evaluate_bp.py")
