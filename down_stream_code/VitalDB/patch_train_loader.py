with open('train_finetune.py', 'r') as f:
    content = f.read()

# Fix subject_ids references in DataLoader calls
content = content.replace(
    'subject_ids[train_idx]',
    'subject_ids_train'
)
content = content.replace(
    'subject_ids[val_idx]',
    'subject_ids_val_arr'
)
content = content.replace(
    'subject_ids[test_idx]',
    'subject_ids_train'   # unused for test in training script
)

with open('train_finetune.py', 'w') as f:
    f.write(content)

print("✅ Fixed subject_ids references")
