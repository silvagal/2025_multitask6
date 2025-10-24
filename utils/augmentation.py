import numpy as np

def augment_signal(signal_np):
    augmented_signal = signal_np.copy()
    if np.random.rand() < 0.5:
        augmented_signal = -augmented_signal
    noise_amp = np.std(augmented_signal) * np.random.uniform(0.01, 0.15)
    if np.std(augmented_signal) == 0:
        noise_amp = np.random.uniform(0.01, 0.05)
    noise = np.random.normal(0, noise_amp, size=augmented_signal.shape)
    augmented_signal += noise
    return augmented_signal