import torch
import numpy as np
import yaml
from diffusion_model_no_compress_final import diff_CSDI, CSDI_base

device = torch.device("cuda")
with open('base_no_compress_original.yaml', 'r') as f:
    config = yaml.safe_load(f)

data   = np.load('/cpfs01/shane/jess_data/data/archive/unicardio_vitaldb_test.npz')
ppg    = torch.tensor(data['ppg'][:32], dtype=torch.float32)
ecg    = torch.tensor(data['ecg'][:32], dtype=torch.float32)
abp    = torch.tensor(data['abp'][:32], dtype=torch.float32)
dummy  = torch.zeros(32, 500)
flat   = torch.cat([ppg, ecg, dummy, abp], dim=1)
signal = flat.unsqueeze(1).to(device)

Model = CSDI_base(config, device, L=500*4).to(device)
Model = torch.nn.DataParallel(Model).to(device)
Model.load_state_dict(torch.load("no_compress799.pth", weights_only=False))
Model.eval()

with torch.no_grad():
    results = Model(signal, n_samples=2, model_flag='03',
                    borrow_mode=2, DDIM_flag=1, sample_steps=6, train_gen_flag=1)

print(f"Type: {type(results)}")
if isinstance(results, tuple):
    for i, r in enumerate(results):
        print(f"  [{i}]: type={type(r)}, shape={r.shape if hasattr(r,'shape') else 'N/A'}")
else:
    print(f"Shape: {results.shape}")
