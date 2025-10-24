import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sys

# --- CONFIGURAÇÃO DE FONTE ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Apenas para fins de demonstração se utils/config não estiverem disponíveis
try:
    from utils.pretext_task import convert_to_pretext_task, generate_pretext_sequences
    from config import QRS_COMPLEX_PROPORTION, ROOT_OUTPUT_DIR, MIT_BEAT_PRE_R, MIT_BEAT_POST_R
except ImportError:
    print("Warning: Could not import from 'utils' or 'config'. Using dummy functions and default values.")
    MIT_BEAT_PRE_R = 108
    MIT_BEAT_POST_R = 108

# --- CONFIGURAÇÕES GLOBAIS DO SCRIPT ---
project_root = os.path.dirname(os.path.abspath(__file__))
ROOT_OUTPUT_DIR = os.path.join(project_root, "output", "plots", "pretext")

QRS_COMPLEX_PROPORTION = 0.20

# --- Funções Auxiliares e de Geração de Dados ---

def convert_to_pretext_task(beats_batch: np.ndarray, qrs_proportion: float):
    if beats_batch.ndim == 1:
        beats_batch = np.expand_dims(beats_batch, axis=0)
    num_beats_in_batch, beat_size = beats_batch.shape
    p_len, qrs_len, t_len = get_segment_lengths(beat_size, qrs_proportion)
    p_segments = beats_batch[:, :p_len]
    qrs_segments = beats_batch[:, p_len:p_len + qrs_len]
    t_segments = beats_batch[:, p_len + qrs_len:]
    permutations = [
        (p_segments, qrs_segments, t_segments), (p_segments, t_segments, qrs_segments),
        (qrs_segments, p_segments, t_segments), (qrs_segments, t_segments, p_segments),
        (t_segments, p_segments, qrs_segments), (t_segments, qrs_segments, p_segments)
    ]
    augmented_beats = np.array([np.concatenate(p, axis=1) for p in permutations])
    augmented_labels = np.arange(6)
    return augmented_beats, augmented_labels

def generate_pretext_sequences(signals_tensor, r_peaks_tensor, main_labels, rr_features, qrs_proportion):
    sequence = signals_tensor.numpy().flatten()
    r_peaks = r_peaks_tensor[0].numpy()
    beat_len = MIT_BEAT_PRE_R + MIT_BEAT_POST_R + 1
    original_beats_in_sequence, r_peak_indices_in_bounds = [], []
    for r_peak in r_peaks:
        start = r_peak - MIT_BEAT_PRE_R
        end = r_peak + MIT_BEAT_POST_R + 1
        if start >= 0 and end <= len(sequence):
            original_beats_in_sequence.append(sequence[start:end])
            r_peak_indices_in_bounds.append(r_peak)
    if not original_beats_in_sequence:
        return torch.tensor([]), None, torch.tensor([]), None
    original_beats_in_sequence = np.array(original_beats_in_sequence)
    permuted_beats_all, labels = convert_to_pretext_task(original_beats_in_sequence, qrs_proportion)
    permuted_sequences_list = []
    for i in range(6):
        new_sequence = sequence.copy()
        current_permutation_of_beats = permuted_beats_all[i]
        for j, r_peak in enumerate(r_peak_indices_in_bounds):
            start = r_peak - MIT_BEAT_PRE_R
            end = r_peak + MIT_BEAT_POST_R + 1
            if start >= 0 and end <= len(new_sequence):
                new_sequence[start:end] = current_permutation_of_beats[j]
        permuted_sequences_list.append(new_sequence)
    return torch.tensor(np.array(permuted_sequences_list)), None, torch.arange(6), None

def process_synthetic_signal(signal: np.ndarray, smoothing_window: int = 7) -> np.ndarray:
    min_val, max_val = np.min(signal), np.max(signal)
    if max_val - min_val > 1e-6:
        signal = 2 * ((signal - min_val) / (max_val - min_val)) - 1
    if smoothing_window > 1:
        window = np.ones(smoothing_window) / smoothing_window
        signal = np.convolve(signal, window, mode='same')
    return signal

def create_synthetic_beat(size=300):
    x = np.linspace(-np.pi, np.pi, size)
    p_wave = 0.25 * np.exp(-((x + 1.8)**2) * 8)
    q_wave = -0.2 * np.exp(-((x + 0.2)**2) * 80)
    r_wave = 1.9 * np.exp(-(x**2) * 200)
    s_wave = -0.45 * np.exp(-((x - 0.18)**2) * 80)
    qrs_complex = q_wave + r_wave + s_wave
    t_wave = 0.55 * np.exp(-((x - 1.5)**2) * 5)
    baseline_wander = 0.05 * np.sin(x * 0.8)
    beat = p_wave + qrs_complex + t_wave + baseline_wander
    noise = np.random.normal(0, 0.04, size)
    return process_synthetic_signal(beat + noise)

def get_segment_lengths(beat_size, qrs_proportion):
    qrs_size = int(beat_size * qrs_proportion)
    p_t_size = beat_size - qrs_size
    p_size = p_t_size // 2
    t_size = p_t_size - p_size
    return p_size, qrs_size, t_size

def create_synthetic_sequence(seq_len=1250, beat_size=300): 
    base_signal = np.random.normal(0, 0.03, seq_len) 
    r_peak_locs = [200, 550, 900] 
    for peak_loc in r_peak_locs: 
        x = np.linspace(-np.pi, np.pi, beat_size) 
        p_wave = 0.2 * np.cos(x*1.5 - 2.2)**2 * np.exp(-x*0.1) 
        qrs_complex = 1.5 * np.exp(-(x*2.5)**2) - 0.4 * np.exp(-((x*2.5 - 0.2)**2) / 0.1) 
        t_wave = 0.5 * np.exp(-((x - 1.5)**2) / 1.0) 
        beat = p_wave + qrs_complex + t_wave 
        noise = np.random.normal(0, 0.05, beat_size) 
        start = peak_loc - beat_size // 2 
        end = start + beat_size 
        if start >= 0 and end <= seq_len: 
            base_signal[start:end] += (beat + noise) 
    return process_synthetic_signal(base_signal), r_peak_locs

# --- Visualization Functions ---

def visualize_single_beat_permutations(beat: np.ndarray, qrs_proportion: float):
    augmented_beats_batch, augmented_labels = convert_to_pretext_task(beat, qrs_proportion)
    augmented_beats = augmented_beats_batch[:, 0, :]
    beat_size = len(beat)
    p_len, qrs_len, t_len = get_segment_lengths(beat_size, qrs_proportion)
    permutation_titles = {
        0: "Original: P-QRS-T", 1: "Permutation: P-T-QRS", 2: "Permutation: QRS-P-T",
        3: "Permutation: QRS-T-P", 4: "Permutation: T-P-QRS", 5: "Permutation: T-QRS-P"
    }
    segment_info = {
        'P': {'len': p_len, 'color': 'skyblue'}, 'QRS': {'len': qrs_len, 'color': 'salmon'},
        'T': {'len': t_len, 'color': 'lightgreen'}
    }
    permutation_order = {
        0: ['P', 'QRS', 'T'], 1: ['P', 'T', 'QRS'], 2: ['QRS', 'P', 'T'],
        3: ['QRS', 'T', 'P'], 4: ['T', 'P', 'QRS'], 5: ['T', 'QRS', 'P']
    }
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    axes = axes.ravel()
    # MODIFICADO: Tamanho da fonte das letras P, QRS, T aumentado
    font_props = {'size': 28, 'weight': 'bold'}
    y_pos_labels = np.max(beat) * 1.15 
    if y_pos_labels < 0.9: y_pos_labels = 0.9
    for i, (ax, signal, label) in enumerate(zip(axes, augmented_beats, augmented_labels)):
        ax.plot(signal, color='black', linewidth=1.5)
        current_pos = 0
        order = permutation_order[label]
        for seg_name in order:
            info = segment_info[seg_name]
            seg_len = info['len']
            ax.axvspan(current_pos, current_pos + seg_len, color=info['color'], alpha=0.3)
            ax.text(current_pos + seg_len / 2, y_pos_labels, seg_name, fontdict=font_props, ha='center', va='center')
            current_pos += seg_len
        # MODIFICADO: Tamanho do título do subplot aumentado
        ax.set_title(permutation_titles.get(label), fontsize=20)
        ax.set_ylim(-1.2, y_pos_labels * 1.1) 
        ax.grid(True, linestyle='--', alpha=0.6)
        if i % 3 == 0:
            # MODIFICADO: Tamanho do rótulo do eixo Y aumentado
            ax.set_ylabel("Normalized Amplitude", fontsize=20)
        # MODIFICADO: Tamanho dos valores dos eixos aumentado
        ax.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = os.path.join(ROOT_OUTPUT_DIR, "pretext_single_beat_permutations_grid.png")
    os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path)
    print(f"Single beat permutations grid visualization saved to: {output_path}")
    plt.close()

def visualize_full_sequence_permutations(sequence: np.ndarray, r_peaks: list, qrs_proportion: float):
    signals_tensor = torch.from_numpy(sequence).float().unsqueeze(0)
    r_peaks_tensor = [torch.tensor(r_peaks)]
    main_labels = torch.tensor([0])
    rr_features = torch.tensor([[0.8]])
    aug_signals, _, aug_pretext_labels, _ = generate_pretext_sequences(
        signals_tensor, r_peaks_tensor, main_labels, rr_features, qrs_proportion
    )
    if aug_signals.numel() == 0:
        print("No valid beats found in sequence for permutation visualization.")
        return
    beat_size = MIT_BEAT_PRE_R + MIT_BEAT_POST_R + 1
    p_len, qrs_len, t_len = get_segment_lengths(beat_size, qrs_proportion)
    permutation_titles = {
        0: "Original: P-QRS-T", 1: "Permutation: P-T-QRS", 2: "Permutation: QRS-P-T",
        3: "Permutation: QRS-T-P", 4: "Permutation: T-P-QRS", 5: "Permutation: T-QRS-P"
    }
    segment_info = {
        'P': {'len': p_len, 'color': 'skyblue'}, 'QRS': {'len': qrs_len, 'color': 'salmon'},
        'T': {'len': t_len, 'color': 'lightgreen'}
    }
    permutation_order = {
        0: ['P', 'QRS', 'T'], 1: ['P', 'T', 'QRS'], 2: ['QRS', 'P', 'T'],
        3: ['QRS', 'T', 'P'], 4: ['T', 'P', 'QRS'], 5: ['T', 'QRS', 'P']
    }
    fig, axes = plt.subplots(3, 2, figsize=(30, 15), sharex=True, sharey=True)
    axes = axes.ravel()
    # MODIFICADO: Tamanho da fonte das letras P, QRS, T aumentado
    font_props = {'size': 22, 'weight': 'bold'}
    y_pos_labels = np.max(sequence) * 1.15
    if y_pos_labels < 0.9: y_pos_labels = 0.9
    for i, (ax, signal_tensor, label_tensor) in enumerate(zip(axes, aug_signals, aug_pretext_labels)):
        signal = signal_tensor.numpy()
        label = label_tensor.item()
        ax.plot(signal, color='black', linewidth=1.5)
        # MODIFICADO: Tamanho do título do subplot aumentado
        ax.set_title(permutation_titles.get(label), fontsize=20)
        r_peak_indices_in_bounds = []
        for r_peak in r_peaks:
            start = r_peak - MIT_BEAT_PRE_R
            end = r_peak + MIT_BEAT_POST_R + 1
            if start >= 0 and end <= len(sequence):
                r_peak_indices_in_bounds.append(r_peak)
        for r_peak in r_peak_indices_in_bounds:
            beat_start_offset = r_peak - MIT_BEAT_PRE_R
            if beat_start_offset < 0: continue
            current_pos_in_beat = 0
            order = permutation_order[label] 
            for seg_name in order:
                info = segment_info[seg_name]
                seg_len = info['len']
                ax.axvspan(beat_start_offset + current_pos_in_beat, 
                           beat_start_offset + current_pos_in_beat + seg_len, 
                           color=info['color'], alpha=0.3)
                ax.text(beat_start_offset + current_pos_in_beat + seg_len / 2, 
                        y_pos_labels, seg_name, fontdict=font_props, ha='center', va='center')
                current_pos_in_beat += seg_len
            ax.axvline(x=beat_start_offset, color='gray', linestyle='--', alpha=0.7)
            ax.axvline(x=beat_start_offset + beat_size, color='gray', linestyle='--', alpha=0.7)
        ax.set_ylim(-1.2, y_pos_labels * 1.1)
        ax.grid(True, linestyle='--', alpha=0.6)
        # MODIFICADO: Tamanho dos valores dos eixos aumentado
        ax.tick_params(axis='both', which='major', labelsize=20)
        if i % 2 == 0:
            # MODIFICADO: Tamanho do rótulo do eixo Y aumentado
            ax.set_ylabel("Normalized Amplitude", fontsize=20)
    plt.tight_layout()
    output_path = os.path.join(ROOT_OUTPUT_DIR, "pretext_full_sequence_visualization.png")
    os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path)
    print(f"Full sequence visualization saved to: {output_path}")
    plt.close()

if __name__ == '__main__':
    try:
        MIT_BEAT_PRE_R
    except NameError:
        MIT_BEAT_PRE_R = 108
        MIT_BEAT_POST_R = 108
    beat_len = MIT_BEAT_PRE_R + MIT_BEAT_POST_R + 1
    sample_beat = create_synthetic_beat(size=beat_len)
    visualize_single_beat_permutations(sample_beat, QRS_COMPLEX_PROPORTION)
    sample_sequence, r_peaks = create_synthetic_sequence(beat_size=beat_len)
    visualize_full_sequence_permutations(sample_sequence, r_peaks, QRS_COMPLEX_PROPORTION)