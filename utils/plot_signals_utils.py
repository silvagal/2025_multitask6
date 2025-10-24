from config import SAMPLING_RATE
import matplotlib.pyplot as plt
import os
import random
from utils.rr_computing import detect_r_peaks_scipy

def plot_signals_with_r_peaks(signals_tensor, labels_tensor, num_samples=5, sampling_rate=SAMPLING_RATE, plot_dir="plots"):
    print(f"\nPlotting {num_samples} signals with detected R-peaks (SciPy find_peaks)...")
    os.makedirs(plot_dir, exist_ok=True)

    if len(signals_tensor) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but only {len(signals_tensor)} are available.")
        num_samples = len(signals_tensor)

    selected_indices = random.sample(range(len(signals_tensor)), num_samples)

    for i, idx in enumerate(selected_indices):
        plt.figure(figsize=(12, 6))
        signal_np = signals_tensor[idx].numpy().squeeze()
        label = labels_tensor[idx].item()

        r_peaks, filtered_signal = [], None
        try:
            r_peaks, filtered_signal = detect_r_peaks_scipy(signal_np, sampling_rate)
        except Exception as e:
            print(f"Could not detect R-peaks for signal {idx} for plotting: {e}")

        # Plot the original signal
        plt.plot(signal_np, label=f'Signal {idx}')

        # If R-peaks were successfully detected, plot them on the original signal
        if r_peaks is not None and len(r_peaks) > 0:
            # Use the original signal's amplitude for the y-coordinate of the R-peak
            plt.plot(r_peaks, signal_np[r_peaks], 'ro', markersize=6, label='R-peaks')

        num_peaks = len(r_peaks) if r_peaks is not None else 0
        plt.title(f'Signal {idx} - Class: {label} (Num R-peaks: {num_peaks})')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        file_name = os.path.join(plot_dir, f"signal_{idx}_r_peaks.png")
        plt.savefig(file_name)
        print(f"Plot for signal {idx} saved to {file_name}")
        plt.close()

def plot_signals_with_annotations(signals_tensor, labels_tensor, r_peaks_list, num_samples=5, plot_dir="plots"):
    print(f"\nPlotting {num_samples} signals with provided annotations...")
    os.makedirs(plot_dir, exist_ok=True)

    if len(signals_tensor) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but only {len(signals_tensor)} are available.")
        num_samples = len(signals_tensor)

    selected_indices = random.sample(range(len(signals_tensor)), num_samples)

    for i, idx in enumerate(selected_indices):
        plt.figure(figsize=(12, 6))
        signal_np = signals_tensor[idx].numpy().squeeze()
        label = labels_tensor[idx].item()
        r_peaks = r_peaks_list[idx].numpy()

        plt.plot(signal_np, label=f'Signal {idx}')
        if r_peaks is not None and len(r_peaks) > 0:
            plt.plot(r_peaks, signal_np[r_peaks], 'ro', markersize=6, label='R-peaks')

        num_peaks = len(r_peaks) if r_peaks is not None else 0
        plt.title(f'Signal {idx} - Class: {label} (Num R-peaks: {num_peaks})')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        file_name = os.path.join(plot_dir, f"signal_{idx}_annotated.png")
        plt.savefig(file_name)
        print(f"Plot for signal {idx} saved to {file_name}")
        plt.close()
