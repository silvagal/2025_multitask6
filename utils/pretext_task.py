import numpy as np
import torch

from config import MIT_BEAT_PRE_R, MIT_BEAT_POST_R

def convert_to_pretext_task(x: np.ndarray, qrs_complex_proportion: float = 0.4):
    """
    Permutes the P, QRS, and T segments of a batch of heartbeats.

    For each beat in the input array `x`, this function generates all 6 possible
    permutations of its three segments.

    Args:
        x (np.ndarray): A numpy array of shape (n_beats, beat_size), where
                        each row is a single heartbeat.
        qrs_complex_proportion (float): The proportion of the beat that the
                                        QRS complex represents.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x_augmented (np.ndarray): The augmented beats array of shape
                                        (n_beats * 6, beat_size, 1).
            - y_augmented (np.ndarray): The corresponding permutation labels (0-5).
    """
    x_augmented = []
    y_augmented = []
    beat_size = x.shape[1]

    qrs_half_size = int(beat_size * qrs_complex_proportion) // 2
    qrs_signal_begin = beat_size // 2 - qrs_half_size
    qrs_signal_end = beat_size // 2 + qrs_half_size

    # Ensure begin and end are within bounds
    qrs_signal_begin = max(0, qrs_signal_begin)
    qrs_signal_end = min(beat_size, qrs_signal_end)

    for xi in x:
        p = xi[:qrs_signal_begin]
        qrs = xi[qrs_signal_begin:qrs_signal_end]
        t = xi[qrs_signal_end:]

        permutations = [
            (np.concatenate([p, qrs, t]), 0),  # P-QRS-T (Original)
            (np.concatenate([p, t, qrs]), 1),  # P-T-QRS
            (np.concatenate([qrs, p, t]), 2),  # QRS-P-T
            (np.concatenate([qrs, t, p]), 3),  # QRS-T-P
            (np.concatenate([t, p, qrs]), 4),  # T-P-QRS
            (np.concatenate([t, qrs, p]), 5),  # T-QRS-P
        ]

        for signal, label in permutations:
            x_augmented.append(signal)
            y_augmented.append(label)

    return np.asarray(x_augmented).reshape((-1, beat_size, 1)), np.asarray(y_augmented)


def generate_pretext_sequences(signals: torch.Tensor, r_peaks: list, main_labels: torch.Tensor, rr_features: torch.Tensor, qrs_proportion: float):
    """
    Generates augmented sequences based on beat permutations for the pretext task.

    For "Normal" signals, it creates 6 permuted versions of the entire sequence.
    For "Arrhythmia" signals, it returns the original signal with a pretext label of 0.

    Args:
        signals (torch.Tensor): The input signals (batch_size, sequence_length).
        r_peaks (list[torch.Tensor]): A list of R-peak indices for each signal.
        main_labels (torch.Tensor): The arrhythmia labels for each signal.
        rr_features (torch.Tensor): The RR interval features for each signal.
        qrs_proportion (float): The proportion of the beat considered as the QRS complex.

    Returns:
        tuple: A tuple containing the augmented signals, main labels, pretext labels, and RR features.
    """
    aug_signals, aug_main_labels, aug_pretext_labels, aug_rr = [], [], [], []

    beat_size = MIT_BEAT_PRE_R + MIT_BEAT_POST_R + 1 # 150 + 149 + 1 = 300

    for i in range(len(signals)):
        signal = signals[i]
        label = main_labels[i]
        peaks = r_peaks[i]
        rr = rr_features[i]

        def add_original_signal():
            """Helper to add the original signal and its labels to the augmented lists."""
            aug_signals.append(signal)
            aug_main_labels.append(label)
            aug_pretext_labels.append(torch.tensor(0, dtype=torch.long)) # Original signal has pretext label 0
            aug_rr.append(rr)

        # Apply pretext task only to normal beats (label 0)
        if label == 0:
            # 1. Extract all beats from the signal
            beats = []
            beat_locations = []
            for peak in peaks:
                start, end = peak - MIT_BEAT_PRE_R, peak + MIT_BEAT_POST_R + 1
                if start >= 0 and end <= len(signal):
                    beats.append(signal[start:end].numpy())
                    beat_locations.append((start, end))

            if not beats: # If no valid beats were found, keep original
                add_original_signal()
                continue

            # 2. Partition each beat into P, QRS, T segments
            qrs_half = int(beat_size * qrs_proportion) // 2
            qrs_begin_idx = beat_size // 2 - qrs_half
            qrs_end_idx = beat_size // 2 + qrs_half

            p_waves = [b[:qrs_begin_idx] for b in beats]
            qrs_complexes = [b[qrs_begin_idx:qrs_end_idx] for b in beats]
            t_waves = [b[qrs_end_idx:] for b in beats]

            # 3. Generate 6 new signals, one for each permutation
            permutations = [
                (p_waves, qrs_complexes, t_waves, 0),  # P-QRS-T
                (p_waves, t_waves, qrs_complexes, 1),  # P-T-QRS
                (qrs_complexes, p_waves, t_waves, 2),  # QRS-P-T
                (qrs_complexes, t_waves, p_waves, 3),  # QRS-T-P
                (t_waves, p_waves, qrs_complexes, 4),  # T-P-QRS
                (t_waves, qrs_complexes, p_waves, 5),  # T-QRS-P
            ]

            for part1, part2, part3, pretext_label in permutations:
                new_signal = signal.clone()
                # Reconstruct the signal by placing the permuted beats back
                for j, (start, end) in enumerate(beat_locations):
                    permuted_beat = np.concatenate([part1[j], part2[j], part3[j]])
                    new_signal[start:end] = torch.from_numpy(permuted_beat)

                aug_signals.append(new_signal)
                aug_main_labels.append(label)
                aug_pretext_labels.append(torch.tensor(pretext_label, dtype=torch.long))
                aug_rr.append(rr)
        else:
            # For arrhythmia signals, do not apply pretext augmentation.
            # Just add the original signal.
            add_original_signal()

    return (torch.stack(aug_signals),
            torch.stack(aug_main_labels),
            torch.stack(aug_pretext_labels),
            torch.stack(aug_rr))