with open('train_finetune.py', 'r') as f:
    lines = f.readlines()

with open('train_finetune.py.bak4', 'w') as f:
    f.writelines(lines)

# Insert DataLoader block before line 257 (0-indexed: 256)
dataloader_block = '''
# ── VitalDB Dataset & DataLoader ──
from torch.utils.data import Dataset, DataLoader

class CustomSignalDataset(Dataset):
    def __init__(self, signal, impute, noisy, mask, subject_ids):
        self.signal = signal
        self.impute = impute
        self.noisy  = noisy
        self.mask   = mask
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, idx):
        return (self.signal[idx], self.impute[idx],
                self.noisy[idx], self.mask[idx],
                self.subject_ids[idx])

train_dataset = CustomSignalDataset(
    inputs_train, impute_train, noisy_train, mask_train, subject_ids_train)
val_dataset   = CustomSignalDataset(
    inputs_val, impute_val, noisy_val, mask_val, subject_ids_val_arr)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          pin_memory=True, num_workers=8)
val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False)

print(f"train_loader: {len(train_loader)} batches | val_loader: {len(val_loader)} batches")

'''

# Insert before line 257 (0-indexed 256)
patched = lines[:255] + [dataloader_block + '\n'] + lines[255:]

with open('train_finetune.py', 'w') as f:
    f.writelines(patched)

print("✅ Added DataLoader block before training loop")
