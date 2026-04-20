with open('run_inference_vitaldb.py', 'r') as f:
    content = f.read()

# Slice all signal tensors to N after loading
content = content.replace(
    'N   = 256  # [TEST] quick eval on 256 samples',
    'N   = ppg.shape[0]'
)
content = content.replace(
    'dummy  = torch.zeros(N, 500)',
    'dummy  = torch.zeros(N, 500)\n# Subset for quick test\nN = 256\nppg=ppg[:N]; ecg=ecg[:N]; abp=abp[:N]; dummy=dummy[:N]'
)

with open('run_inference_vitaldb.py', 'w') as f:
    f.write(content)
print("✅ Fixed")
