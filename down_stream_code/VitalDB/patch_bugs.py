with open('train_finetune.py', 'r') as f:
    content = f.read()

with open('train_finetune.py.bak5', 'w') as f:
    f.write(content)

fixes = [
    # Bug 1 — 3D unpack on 2D tensor
    ('N, channels, L = signal.shape',
     'N = signal.shape[0]  # [VitalDB] signal is (N, 2000)'),

    # Bug 2 — double tensor wrapping
    ('signal = torch.tensor(signal, dtype = torch.float32)',
     '# [VitalDB] signal already a tensor from load_vitaldb()'),

    # Bug 3 — mask from undefined signal_impute
    ('mask = mask[:,0,:]',
     'mask = torch.zeros([signal.shape[0], signal.shape[1] // 4])  # [VitalDB] dummy mask'),

    # Bug 4 — reshape on already-flat tensor
    ('signal = torch.reshape(signal, [signal.shape[0], 3*signal.shape[2]])',
     '# [VitalDB] signal already flat (N, 2000) — no reshape needed'),
]

for old, new in fixes:
    if old in content:
        content = content.replace(old, new)
        print(f"✅ Fixed: {old[:50]}...")
    else:
        print(f"⚠️  Not found: {old[:50]}...")

with open('train_finetune.py', 'w') as f:
    f.write(content)

print("\nDone — run: sed -n '200,230p' train_finetune.py to verify")
