import numpy as np

# Load your preprocessed data
data = np.load('/cpfs01/shane/jess_data/data/archive/unicardio_ready.npz')
ecg = data['ecg']   # (57600, 500)
ppg = data['ppg']   # (57600, 500)
abp = data['abp']   # (57600, 500)

N = ecg.shape[0]

# Stack into shape (N, 3, 500)
# Repo channel order: [0]=PPG, [1]=ABP, [2]=ECG
signal = np.zeros((N, 3, 500), dtype=np.float32)
signal[:, 0, :] = ppg
signal[:, 1, :] = abp
signal[:, 2, :] = ecg

# Normalize ABP: repo does (x - 100) / 50
signal[:, 1, :] = (signal[:, 1, :] - 100) / 50

print(f"Final shape: {signal.shape}")
print(f"ABP normalized range: {signal[:,1,:].min():.2f} to {signal[:,1,:].max():.2f}")

# Save
np.save('base_model/Final_sig_combined.npy', signal)
print("Saved to base_model/Final_sig_combined.npy")

