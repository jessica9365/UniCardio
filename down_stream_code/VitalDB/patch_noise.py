with open('train_finetune.py', 'r') as f:
    lines = f.readlines()

with open('train_finetune.py.bak3', 'w') as f:
    f.writelines(lines)

patched = []
skip_keywords = ['signal_impute', 'signal_noisy', 'AddNoise', 'imputation_pattern',
                 'null = torch', 'torch.concatenate([signal', 'torch.concatenate([signal_impute',
                 'torch.concatenate([signal_noisy']
for line in lines:
    if any(kw in line for kw in skip_keywords):
        patched.append('# [VitalDB] ' + line)  # comment out, don't delete
    else:
        patched.append(line)

with open('train_finetune.py', 'w') as f:
    f.writelines(patched)

print("✅ Commented out imputation/noise lines")
