import numpy as np
from scipy.signal import find_peaks, medfilt, butter, filtfilt
from tqdm import tqdm
import torch

class Scaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None
    def fit(self, data_tensor):
        self.min_val = data_tensor.min().item()
        self.max_val = data_tensor.max().item()
    def transform(self, data_tensor):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        if (self.max_val - self.min_val) == 0:
            return torch.zeros_like(data_tensor) if self.min_val == 0 else data_tensor / self.min_val
        return (data_tensor - self.min_val) / (self.max_val - self.min_val + 1e-8)
    def fit_transform(self, data_tensor):
        self.fit(data_tensor)
        return self.transform(data_tensor)
    def inverse_transform(self, data_tensor_normalized):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted yet.")
        return data_tensor_normalized * (self.max_val - self.min_val + 1e-8) + self.min_val

def detect_r_peaks_scipy(signal, sampling_rate):
    lowcut = 0.5
    highcut = 40.0
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    squared_signal = filtered_signal**2
    smoothed_signal = medfilt(squared_signal, kernel_size=int(0.08 * sampling_rate) | 1)
    peaks, _ = find_peaks(smoothed_signal, distance=int(0.2 * sampling_rate), prominence=np.std(smoothed_signal) * 0.5)
    refined_peaks = []
    window_size = int(0.05 * sampling_rate)
    for peak_idx in peaks:
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(filtered_signal), peak_idx + window_size)
        window = filtered_signal[start_idx:end_idx]
        if len(window) > 0:
            max_in_window_offset = np.argmax(window)
            refined_peaks.append(start_idx + max_in_window_offset)
    return np.array(refined_peaks, dtype=int), filtered_signal

def compute_rr_features(signals_tensor_list, sampling_rate, rr_scaler=None, fit_scalers=False):
    rr_intervals_list = []
    for sig_tensor in tqdm(signals_tensor_list, desc="Computing RR Intervals (SciPy find_peaks)"):
        sig_np = sig_tensor.numpy().squeeze()
        r_peaks, _ = detect_r_peaks_scipy(sig_np, sampling_rate)
        if len(r_peaks) >= 2:
            rr_values = np.diff(r_peaks)
            rr_intervals_list.append(np.mean(rr_values) if len(rr_values) > 0 else 0.0)
        else:
            rr_intervals_list.append(0.0)
    rr_tensor = torch.tensor(rr_intervals_list, dtype=torch.float32)
    current_rr_scaler = rr_scaler if rr_scaler else Scaler()
    if fit_scalers:
        rr_tensor_normalized = current_rr_scaler.fit_transform(rr_tensor)
    else:
        if not (current_rr_scaler.min_val is not None):
            raise ValueError("RR Scaler must be fitted or provided for transformation.")
        rr_tensor_normalized = current_rr_scaler.transform(rr_tensor)
    return rr_tensor_normalized, current_rr_scaler

def compute_rr_features_from_annotations(r_peaks_list, rr_scaler=None, fit_scalers=False):
    rr_intervals_list = []
    for r_peaks in tqdm(r_peaks_list, desc="Computing RR Intervals from Annotations"):
        if len(r_peaks) >= 2:
            rr_values = np.diff(r_peaks.numpy())
            rr_intervals_list.append(np.mean(rr_values) if len(rr_values) > 0 else 0.0)
        else:
            rr_intervals_list.append(0.0)
    rr_tensor = torch.tensor(rr_intervals_list, dtype=torch.float32)
    current_rr_scaler = rr_scaler if rr_scaler else Scaler()
    if fit_scalers:
        rr_tensor_normalized = current_rr_scaler.fit_transform(rr_tensor)
    else:
        if not (current_rr_scaler.min_val is not None):
            raise ValueError("RR Scaler must be fitted or provided for transformation.")
        rr_tensor_normalized = current_rr_scaler.transform(rr_tensor)
    return rr_tensor_normalized, current_rr_scaler