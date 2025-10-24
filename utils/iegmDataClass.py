import numpy as np
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, signals_tensor, labels_tensor, rr_intervals_tensor=None, pretext_labels_tensor=None):
        self.signals = signals_tensor
        self.labels = labels_tensor
        self.rr_intervals = rr_intervals_tensor
        self.pretext_labels = pretext_labels_tensor

        self.has_rr = rr_intervals_tensor is not None
        self.has_pretext = pretext_labels_tensor is not None

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        data = {'IEGM_seg': signal, 'ac': int(self.labels[idx])}

        if self.has_rr:
            data['rr'] = self.rr_intervals[idx]

        if self.has_pretext:
            data['pretext'] = self.pretext_labels[idx]

        return data
    
def normalize_signal(signal_np):
    mean = np.mean(signal_np)
    std = np.std(signal_np)
    if std == 0:
        return signal_np - mean
    return (signal_np - mean) / std