import os
import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm
import sys
from scipy.signal import savgol_filter

# Adicionando o caminho para os módulos locais
# Certifique-se de que o caminho relativo esteja correto para sua estrutura de projeto

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.rr_computing import Scaler
from config import (DATASETS_ROOT)

# MIT Dataset Configuration
MIT_RAW_DATA_PATH = os.path.join(DATASETS_ROOT, "MIT")
MIT_TRAIN_SIGNALS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
MIT_TEST_SIGNALS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
MIT_NORMAL_BEAT_TYPES = ['N', 'L', 'R', 'e', 'j']
MIT_S_BEAT_TYPES = ['A', 'a', 'J', 'S']
MIT_V_BEAT_TYPES = ['V', 'E']
MIT_SEGMENT_SIZE = 1250
MIT_BEAT_PRE_R = 150
MIT_BEAT_POST_R = 149




def preprocess_and_normalize_segment(segment: np.ndarray) -> np.ndarray:
    """
    Aplica o filtro Savitzky-Golay para suavização e depois normaliza para [-1, 1].
    """
    window_length = 37
    polyorder = 2

    # Garante que o window_length seja ímpar e menor que o tamanho do segmento
    if len(segment) > window_length:
        smoothed_segment = savgol_filter(segment, window_length, polyorder)
    else:
        smoothed_segment = segment

    min_val = np.min(smoothed_segment)
    max_val = np.max(smoothed_segment)

    # Evita divisão por zero se o segmento for constante
    if max_val - min_val > 1e-6:
        return 2 * (smoothed_segment - min_val) / (max_val - min_val) - 1
    return smoothed_segment - np.mean(smoothed_segment) # Centraliza em zero se for constante

def _load_signal(ecg_file_id: int):
    """
    Carrega um sinal e suas anotações de um arquivo .mat.
    """
    ecg_path = os.path.join(MIT_RAW_DATA_PATH, f'{ecg_file_id}.mat')
    try:
        struct = loadmat(ecg_path)
        # Tenta a primeira estrutura de chave ('val')
        try:
            data = struct['val'][0]
            signal = data[0].flatten()
            annotations = data[1]
            r_peaks = annotations['sample'][0][0].flatten().astype(int)
            beat_types = annotations['type'][0][0].flatten()
        # Se falhar, tenta a segunda estrutura de chave ('individual')
        except KeyError:
            data = struct['individual'][0][0]
            if ecg_file_id == 114:
                signal = data['signal_r'][:, 1]
            else:
                signal = data['signal_r'][:, 0]
            r_peaks = data['anno_anns'].flatten().astype(int)
            beat_types = data['anno_type'].flatten()

        return signal, r_peaks, beat_types
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {ecg_path}")
        return None, None, None
    except Exception as e:
        print(f"Erro ao carregar {ecg_file_id}.mat: {e}")
        return None, None, None

def process_partition(signal_ids: set, partition_name: str):
    """
    Processa todos os sinais de uma determinada partição (treino ou teste).
    """
    all_segments = []
    all_labels = []
    all_rr_intervals = []
    all_r_peaks_relative = []

    for signal_file in tqdm(sorted(list(signal_ids)), desc=f"Processando Sinais de {partition_name}"):
        signal, r_peaks, beat_types = _load_signal(signal_file)

        if signal is None or len(r_peaks) == 0:
            continue

        r_peak_cursor = 0
        while r_peak_cursor < len(r_peaks):
            current_r_peak_location = r_peaks[r_peak_cursor]

            start_index = current_r_peak_location - (MIT_SEGMENT_SIZE // 2)
            end_index = start_index + MIT_SEGMENT_SIZE

            if start_index < 0 or end_index > len(signal):
                r_peak_cursor += 1
                continue

            segment_r_peaks_indices = np.where((r_peaks >= start_index) & (r_peaks < end_index))[0]

            if len(segment_r_peaks_indices) == 0:
                r_peak_cursor += 1
                continue

            # Garante que o cursor avance para evitar loops infinitos
            last_r_peak_in_segment_idx = segment_r_peaks_indices[-1]
            r_peak_cursor = last_r_peak_in_segment_idx + 1

            segment = np.copy(signal[start_index:end_index])
            processed_segment = preprocess_and_normalize_segment(segment)

            r_peaks_in_segment = r_peaks[segment_r_peaks_indices]
            relative_peaks = r_peaks_in_segment - start_index

            segment_beat_types = beat_types[segment_r_peaks_indices]

            is_normal = all(bt in MIT_NORMAL_BEAT_TYPES for bt in segment_beat_types)
            has_s_or_v = any(bt in MIT_S_BEAT_TYPES or bt in MIT_V_BEAT_TYPES for bt in segment_beat_types)

            if has_s_or_v:
                label = 1
            elif is_normal:
                label = 0
            else:
                continue

            rr_intervals = np.diff(r_peaks_in_segment)
            mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.0

            all_segments.append(torch.tensor(processed_segment, dtype=torch.float32).unsqueeze(0)) # Adiciona dimensão de canal
            all_labels.append(label)
            all_rr_intervals.append(mean_rr)
            all_r_peaks_relative.append(torch.tensor(relative_peaks, dtype=torch.long))

    return all_segments, all_labels, all_rr_intervals, all_r_peaks_relative

def save_data(output_dir: str, partition_name: str, data: tuple, rr_scaler: Scaler = None):
    """
    Salva os tensores processados no disco.
    """
    segments, labels, rr_intervals, r_peaks_relative = data

    if not segments:
        print(f"Nenhum segmento gerado para a partição {partition_name}. Nada será salvo.")
        return

    # Converte listas para tensores
    signals_tensor = torch.cat(segments, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    rr_intervals_tensor = torch.tensor(rr_intervals, dtype=torch.float32) # Remove .view(-1, 1)

    # Normaliza os intervalos RR
    if rr_scaler:
        rr_intervals_normalized = rr_scaler.transform(rr_intervals_tensor)
    else: # Assume que é o conjunto de treino se nenhum scaler for passado
        rr_scaler = Scaler()
        rr_intervals_normalized = rr_scaler.fit_transform(rr_intervals_tensor)
        torch.save(rr_scaler, os.path.join(output_dir, 'mit_rr_scaler.pth'))
        print("Scaler dos intervalos RR treinado e salvo.")

    # Salva os arquivos
    torch.save(signals_tensor, os.path.join(output_dir, f'mit_{partition_name}_signals.pt'))
    torch.save(labels_tensor, os.path.join(output_dir, f'mit_{partition_name}_labels.pt'))
    torch.save(rr_intervals_normalized, os.path.join(output_dir, f'mit_{partition_name}_rr_norm.pt'))
    torch.save(r_peaks_relative, os.path.join(output_dir, f'mit_{partition_name}_r_peaks.pt'))

    print(f"Dados da partição '{partition_name}' salvos em {output_dir}")
    print(f"Total de segmentos: {len(segments)}")
    if len(labels) > 0:
        counts = np.bincount(labels)
        print(f"Distribuição de classes: {counts}")

    return rr_scaler

def main():
    """
    Função principal para orquestrar o pré-processamento dos dados de treino e teste.
    """
    output_directory = os.path.join(BASE_PATH, 'mit')
    os.makedirs(output_directory, exist_ok=True)

    # 1. Processar dados de treino
    train_data = process_partition(set(MIT_TRAIN_SIGNALS), "train")

    # 2. Salvar dados de treino e treinar o scaler
    rr_scaler = save_data(output_directory, "train", train_data)

    # 3. Processar dados de teste
    test_data = process_partition(set(MIT_TEST_SIGNALS), "test")

    # 4. Salvar dados de teste usando o scaler JÁ TREINADO
    if rr_scaler:
        save_data(output_directory, "test", test_data, rr_scaler)
    else:
        print("Aviso: Scaler não foi treinado pois não houve dados de treino. Os dados de teste não serão salvos.")


if __name__ == '__main__':
    main()