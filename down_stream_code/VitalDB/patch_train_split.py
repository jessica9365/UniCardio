with open('train_finetune.py', 'r') as f:
    lines = f.readlines()

with open('train_finetune.py.bak2', 'w') as f:
    f.writelines(lines)

# New block to replace lines 237-296 (0-indexed 236-295)
new_block = '''# ── VitalDB: splits already pre-made, no GroupShuffleSplit needed ──
# signal      = train split (45983, 2000)
# signal_val  = val split   (5747,  2000)
# For BP translation: no imputation or noise needed — pass signal as all inputs

inputs_train = signal
inputs_val   = signal_val

# Dummy impute/noisy/mask — same as clean signal (no masking for translation task)
impute_train = signal.clone()
impute_val   = signal_val.clone()
noisy_train  = signal.clone()
noisy_val    = signal_val.clone()
mask_train   = torch.zeros(signal.shape[0], signal.shape[1] // 4)
mask_val     = torch.zeros(signal_val.shape[0], signal_val.shape[1] // 4)
subject_ids_train = np.arange(len(signal))
subject_ids_val_arr = np.arange(len(signal_val))

'''

patched = lines[:236] + [new_block + '\n'] + lines[296:]

with open('train_finetune.py', 'w') as f:
    f.writelines(patched)

print("✅ Patched lines 237-296 — original saved as train_finetune.py.bak2")
