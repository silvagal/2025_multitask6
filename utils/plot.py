from config import SAMPLING_RATE
import matplotlib.pyplot as plt
import os
import random
from utils.rr_computing import detect_r_peaks_scipy
import seaborn as sns
import numpy as np
import itertools

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

def plot_confusion_matrix(cm, class_names, output_path, title='Confusion Matrix'):
    """
    Renders and saves a confusion matrix as a heatmap.
    """
    # Normalize the confusion matrix.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

def plot_training_history(history, output_dir, file_prefix=''):
    """
    Plots and saves the training and validation loss and accuracy, each to a separate file.
    Handles both single-task and multi-task histories.
    """
    metrics_to_plot = {
        'acc': 'Accuracy',
        'loss_main': 'Main Task Loss',
        'loss_rr': 'Auxiliary (RR) Task Loss'
    }

    for metric, title in metrics_to_plot.items():
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        # Skip plotting if the metric doesn't exist in history (e.g., loss_rr in single-task)
        if train_key not in history or not any(history[train_key]):
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(history[train_key], label='Train')
        plt.plot(history[val_key], label='Validation')
        plt.title(title)
        plt.ylabel(metric.split('_')[-1].capitalize())
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{file_prefix}history_{metric}.png")
        plt.savefig(output_path)
        print(f"Training history plot saved to {output_path}")
        plt.close()

def plot_comparison_history(histories, output_path_prefix):
    """
    Plots and saves comparison graphs for training histories from multiple experiments.
    `histories` is a dict like {'model_name_1': history_1, 'model_name_2': history_2}
    Generates separate plots for train and validation curves.
    """
    metric_keys = ['acc', 'loss_main', 'loss_rr']

    for metric in metric_keys:
        # --- Plot for Training ---
        plt.figure(figsize=(10, 6))
        colors = itertools.cycle(['b', 'r', 'g', 'c', 'm', 'y', 'k'])
        has_train_data = False
        for model_name, history in histories.items():
            train_key = f'train_{metric}'
            if train_key in history and any(history[train_key]):
                plt.plot(history[train_key], label=model_name, color=next(colors))
                has_train_data = True

        if has_train_data:
            plt.title(f"Training {metric.replace('_', ' ').title()} Comparison")
            plt.ylabel(metric.split('_')[-1].capitalize())
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            output_path = f"{output_path_prefix}_train_{metric}.png"
            plt.savefig(output_path)
            print(f"Comparison plot saved to {output_path}")
        plt.close()

        # --- Plot for Validation ---
        plt.figure(figsize=(10, 6))
        colors = itertools.cycle(['b', 'r', 'g', 'c', 'm', 'y', 'k'])
        has_val_data = False
        for model_name, history in histories.items():
            val_key = f'val_{metric}'
            if val_key in history and any(history[val_key]):
                plt.plot(history[val_key], label=model_name, color=next(colors))
                has_val_data = True

        if has_val_data:
            plt.title(f"Validation {metric.replace('_', ' ').title()} Comparison")
            plt.ylabel(metric.split('_')[-1].capitalize())
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            output_path = f"{output_path_prefix}_validation_{metric}.png"
            plt.savefig(output_path)
            print(f"Comparison plot saved to {output_path}")
        plt.close()