import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from scipy.signal import savgol_filter
# Supondo que o seu arquivo process_mit.py esteja na pasta 'mit'
from database.mit.process_mit import preprocess_and_normalize_segment

# --- Configurações Globais ---
MIT_BEAT_PRE_R = 150
MIT_BEAT_POST_R = 150
MIT_SEGMENT_SIZE = 1250

# Configuração global da fonte para a serifada padrão do sistema
rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = False

def generate_synthetic_data(num_seconds=30, sampling_rate=360, heart_rate=75):
    """
    Gera um sinal de ECG sintético e a localização dos picos R.
    Esta versão usa a forma de onda original que causa o leve desalinhamento,
    que será corrigido visualmente no momento do plot.
    """
    print("Gerando dados de ECG sintéticos...")
    
    # Usando a função original de criação de batimentos
    def create_qrs_complex(samples=40):
        qrs = np.zeros(samples)
        r_peak_pos = samples // 2
        qrs[r_peak_pos-2:r_peak_pos+3] = [0.3, 1.0, 1.0, 0.3, 0]
        qrs[r_peak_pos-7:r_peak_pos-2] = [-0.1, -0.2, -0.15, -0.1, -0.05]
        qrs[r_peak_pos+3:r_peak_pos+8] = [-0.15, -0.3, -0.1, -0.05, 0]
        return qrs

    num_samples = num_seconds * sampling_rate
    signal = np.zeros(num_samples)
    beat_interval = int(sampling_rate / (heart_rate / 60.0))
    r_peak_locations = np.arange(beat_interval, num_samples - beat_interval, beat_interval)
    variability = np.random.randint(-10, 10, size=r_peak_locations.shape[0])
    r_peaks = r_peak_locations + variability
    
    qrs_template = create_qrs_complex(samples=40)
    template_center_offset = len(qrs_template) // 2

    for r_peak in r_peaks:
        start = r_peak - template_center_offset
        end = r_peak + template_center_offset
        if start >= 0 and end <= num_samples:
            signal[start:end] += qrs_template
            
    noise = np.random.normal(0, 0.05, num_samples)
    baseline_wander = 0
    
    final_signal = signal + noise + baseline_wander
    
    print("Dados sintéticos gerados com sucesso.")
    return final_signal, r_peaks

def find_segmentation_examples(signal, r_peaks):
    """Encontra um segmento 'bom' e um 'ruim' nos dados."""
    good_example, bad_example = None, None

    for r_peak_location in reversed(r_peaks):
        start_index = r_peak_location - MIT_BEAT_PRE_R
        end_index = start_index + MIT_SEGMENT_SIZE
        if start_index > 0 and end_index < len(signal):
            segment_r_peaks_indices = np.where((r_peaks >= start_index) & (r_peaks < end_index))[0]
            if len(segment_r_peaks_indices) >= 2:
                last_r_peak_in_segment_loc = r_peaks[segment_r_peaks_indices[-1]]
                if (last_r_peak_in_segment_loc + MIT_BEAT_POST_R) > end_index:
                    bad_example = {"segment_signal": signal[start_index:end_index], "segment_r_peaks": r_peaks[segment_r_peaks_indices] - start_index}
                    break
    
    for r_peak_location in r_peaks:
        start_index = r_peak_location - MIT_BEAT_PRE_R
        end_index = start_index + MIT_SEGMENT_SIZE
        if start_index > 0 and end_index < len(signal):
            segment_r_peaks_indices = np.where((r_peaks >= start_index) & (r_peaks < end_index))[0]
            if len(segment_r_peaks_indices) >= 2:
                last_r_peak_in_segment_loc = r_peaks[segment_r_peaks_indices[-1]]
                if (last_r_peak_in_segment_loc + MIT_BEAT_POST_R) < end_index:
                    good_example = {"segment_signal": signal[start_index:end_index], "segment_r_peaks": r_peaks[segment_r_peaks_indices] - start_index}
                    break

    return good_example, bad_example

def correct_peak_positions(signal, r_peaks, search_radius=10):
    """
    Ajusta a posição de cada pico R para o ponto máximo local na sua vizinhança.
    """
    corrected_peaks = []
    for peak in r_peaks:
        start = max(0, peak - search_radius)
        end = min(len(signal), peak + search_radius)
        
        # Procura o ponto máximo na janela
        window = signal[start:end]
        if len(window) > 0:
            local_max_idx = np.argmax(window)
            corrected_peaks.append(start + local_max_idx)
        else:
            corrected_peaks.append(peak) # Mantém o original se a janela for inválida
            
    return np.array(corrected_peaks)

def plot_incomplete_segment(segment_data, output_path):
    """Plota um segmento com um batimento incompleto, antes e depois do processamento."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 9))
    plt.style.use('seaborn-v0_8-whitegrid')

    unprocessed_signal = segment_data['segment_signal']
    r_peaks_original_rel = segment_data['segment_r_peaks']

    # <<< CORREÇÃO: Picos R ajustados para o máximo local ANTES do processamento >>>
    r_peaks_rel = correct_peak_positions(unprocessed_signal, r_peaks_original_rel)

    signal_raw = preprocess_and_normalize_segment(unprocessed_signal)
    
    first_peak_pos = r_peaks_rel[0]
    last_complete_peak_pos = r_peaks_rel[-2] 
    last_incomplete_peak_pos = r_peaks_rel[-1] 

    end_of_last_complete_beat = last_complete_peak_pos + MIT_BEAT_POST_R

    ax1 = axs[0]
    ax1.plot(signal_raw, label='ECG signal', color='black', lw=1.2)
    ax1.plot(r_peaks_rel, signal_raw[r_peaks_rel], 'o', color='green', markersize=6, label='R-Peaks')
    ax1.add_patch(Rectangle((last_incomplete_peak_pos, -1.1), len(signal_raw) - last_incomplete_peak_pos, 2.2, color='red', alpha=0.2, label='Tail to be Removed'))
    ax1.axvline(x=end_of_last_complete_beat, color='orange', linestyle='--', lw=2, label='End of last complete beat')
    
    ax1.add_patch(Rectangle((0, -1.1), first_peak_pos, 2.2, color='blue', alpha=0.15, ec=None))
    
    ax1.text(first_peak_pos / 2, 0.8, f'{MIT_BEAT_PRE_R} points', ha='center', color='blue', fontsize=10, weight='bold', fontfamily='serif')
    ax1.text(last_incomplete_peak_pos + (len(signal_raw) - last_incomplete_peak_pos) / 2, -0.9,
             f'< {MIT_BEAT_POST_R} points', 
             ha='center', va='bottom', color='darkred', fontsize=10, weight='bold', rotation=90, fontfamily='serif')
    
    ax1.set_ylabel('Normalized Amplitude', fontsize=14)
    ax1.set_ylim(-1.1, 1.1)

    ax2 = axs[1]
    
    signal_processed = signal_raw.copy()
    signal_processed[end_of_last_complete_beat:] = 0
    kept_peaks = r_peaks_rel[r_peaks_rel < end_of_last_complete_beat]

    ax2.plot(signal_processed, label='ECG signal', color='black', lw=1.2)
    ax2.plot(kept_peaks, signal_processed[kept_peaks], 'o', color='green', markersize=6, label='Kept R-Peaks')
    ax2.add_patch(Rectangle((end_of_last_complete_beat, -1.1), len(signal_raw) - end_of_last_complete_beat, 2.2, color='red', alpha=0.3, label='Zeroed Region'))
    
    ax2.text((end_of_last_complete_beat + len(signal_raw)) / 2, 0.75, 'Beat\nremoved', ha='center', va='center', color='darkred', fontsize=12, weight='bold', fontfamily='serif')

    ax2.set_ylabel('Normalized Amplitude', fontsize=14)
    ax2.set_ylim(-1.1, 1.1)
    
    for ax in axs:
        ax.set_xlim(0, len(signal_raw) - 1)
        ax.tick_params(axis='both', labelsize=14)

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=4, fancybox=True, fontsize=14, frameon=True, edgecolor='black', shadow=False)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=3, fancybox=True, fontsize=14, frameon=True, edgecolor='black', shadow=False)
    
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def plot_complete_segment(segment_data, output_path):
    """Plota um segmento onde todos os batimentos estão completos."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5))
    plt.style.use('seaborn-v0_8-whitegrid')

    unprocessed_signal = segment_data['segment_signal']
    r_peaks_original_rel = segment_data['segment_r_peaks']
    
    # <<< CORREÇÃO: Picos R ajustados para o máximo local ANTES do processamento >>>
    r_peaks_rel = correct_peak_positions(unprocessed_signal, r_peaks_original_rel)

    signal = preprocess_and_normalize_segment(unprocessed_signal)
    
    first_peak_pos = r_peaks_rel[0]
    last_peak_pos = r_peaks_rel[-1]

    ax.plot(signal, label='ECG signal', color='black', lw=1.2)
    ax.plot(r_peaks_rel, signal[r_peaks_rel], 'o', color='green', markersize=6, label='R-Peaks')

    ax.add_patch(Rectangle((0, -1.1), first_peak_pos, 2.2, color='blue', alpha=0.15, ec=None))
    
    ax.text(first_peak_pos / 2, 0.8, f'{MIT_BEAT_PRE_R} points', ha='center', color='blue', fontsize=10, weight='bold', fontfamily='serif')

    post_context_start = last_peak_pos
    post_context_end = last_peak_pos + MIT_BEAT_POST_R
    ax.add_patch(Rectangle((post_context_start, -1.1), MIT_BEAT_POST_R, 2.2, color='blue', alpha=0.15, ec=None))
    
    ax.text((post_context_start + post_context_end) / 2, 0.8, f'{MIT_BEAT_POST_R} points', ha='center', color='blue', fontsize=10, weight='bold', fontfamily='serif')

    ax.set_ylabel('Normalized Amplitude', fontsize=14)
    ax.set_xlim(0, len(signal) - 1)
    ax.set_ylim(-1.1, 1.1)
    ax.tick_params(axis='both', labelsize=14)
    
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncols=2, fancybox=True, fontsize=14, frameon=True, edgecolor='black', shadow=False)
    
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output/plots/segmentation')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Diretório de saída configurado para: {output_dir}")

    signal, r_peaks = generate_synthetic_data(num_seconds=60, sampling_rate=360, heart_rate=80)
    saved_files = []
    
    print()

    if signal is not None:
        good_seg, bad_seg = find_segmentation_examples(signal, r_peaks)

        if bad_seg:
            filename = 'segmentacao_batimento_incompleto.png'
            output_file_path = os.path.join(output_dir, filename)
            
            print(f"Exemplo de segmento incompleto encontrado. Gerando plot: {filename}")
            plot_incomplete_segment(bad_seg, output_file_path)
            saved_files.append(output_file_path)
        else:
            print("Não foi possível encontrar um segmento com batimento incompleto nos dados sintéticos.")

        if good_seg:
            filename = 'segmentacao_batimento_completo.png'
            output_file_path = os.path.join(output_dir, filename)

            print(f"Exemplo de segmento completo encontrado. Gerando plot: {filename}")
            plot_complete_segment(good_seg, output_file_path)
            saved_files.append(output_file_path)
        else:
            print(f"Não foi possível encontrar um segmento completo nos dados sintéticos.")
            
        print("\n" + "="*50)
        print("Execução finalizada.".center(50))
        print("="*50)

        if saved_files:
            print("\nOs seguintes plots foram salvos:")
            for file_path in saved_files:
                print(f"-> {file_path}")
        else:
            print("\nNenhum plot foi salvo.")
    else:
        print("Falha ao gerar os dados sintéticos.")