import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import sys
from collections import defaultdict
import argparse
import shutil
from sklearn.metrics import roc_curve, roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_NAMES, ROOT_OUTPUT_DIR

def setup_plot_style():
    """Define a estética dos gráficos para publicação."""
    try:
        # Tenta usar uma fonte serifada comum, com fallback para a família serif genérica.
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times', 'Palatino', 'Charter', 'serif']
    except Exception as e:
        print(f"Aviso: Não foi possível definir a fonte serifada. Usando o padrão do Matplotlib. Erro: {e}")

    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 24

def parse_metrics_file(filepath):
    """Analisa um arquivo test_metrics.txt para extrair métricas e outros dados."""
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return None

    patterns = {
        'seed': r"Seed:\s*(\d+)",
        'best_val_loss': r"Best Validation Loss:\s*([\d.eE+-]+)",
        'accuracy': r"Accuracy:\s*([\d.eE+-]+)",
        'precision': r"Precision:\s*([\d.eE+-]+)",
        'recall': r"Recall:\s*([\d.eE+-]+)",
        'f1': r"F1-Score:\s*([\d.eE+-]+)",
        'specificity': r"Specificity:\s*([\d.eE+-]+)",
        'exec_time': r"Execution Time \(s\):\s*([\d.eE+-]+)",
        'total_test_samples': r"Total_Test_Samples:\s*(\d+)",
        'ds1_before': r"DS1_before_downsampling:\s*({.*?})",
        'ds1_after': r"DS1_after_downsampling:\s*({.*?})",
        'ds1_pretext': r"DS1_after_ECGWavePuzzle:\s*({.*?})",
        'ds2_test': r"DS2_test_data:\s*({.*?})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.DOTALL)
        if match:
            val = match.group(1)
            try:
                if '{' in val:
                    metrics[key] = json.loads(val.replace("'", "\""))
                else:
                    metrics[key] = float(val)
            except (json.JSONDecodeError, ValueError):
                metrics[key] = val
    return metrics

def aggregate_and_find_best(experiment_set_dir, seeds):
    """
    Agrega resultados de todos os experimentos, identifica os melhores modelos
    e copia seus artefatos para um diretório central.
    """
    print("\n--- Agregando resultados e identificando melhores modelos ---")
    results = defaultdict(lambda: defaultdict(list))
    histories = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    best_models_info = defaultdict(dict)
    # O diretório best_models agora está dentro do diretório do conjunto de experimentos
    best_models_dir = os.path.join(experiment_set_dir, 'best_models')
    os.makedirs(best_models_dir, exist_ok=True)

    experiment_types = ["baseline", "2_task_multitask", "3_task_multitask"]

    for model_name in MODEL_NAMES:
        # Cria um subdiretório para cada modelo dentro de best_models
        model_output_dir = os.path.join(best_models_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        for exp_type in experiment_types:
            best_seed, min_val_loss, best_run_path = None, float('inf'), None

            for seed in seeds:
                run_path = os.path.join(experiment_set_dir, model_name, str(seed), exp_type)
                metrics_file = os.path.join(run_path, "test_metrics.txt")

                if os.path.exists(metrics_file):
                    metrics = parse_metrics_file(metrics_file)
                    if metrics:
                        results[model_name][exp_type].append(metrics)

                        if 'best_val_loss' in metrics and metrics['best_val_loss'] < min_val_loss:
                            min_val_loss = metrics['best_val_loss']
                            best_seed = int(metrics['seed'])
                            best_run_path = run_path

                history_file = os.path.join(run_path, "history.pt")
                if os.path.exists(history_file):
                    history = torch.load(history_file, weights_only=False)
                    histories[model_name][exp_type]['main'].append(np.array(history['train_loss_main']))
                    total_loss = np.array(history['train_loss_main'])
                    if 'train_loss_rr' in history and history['train_loss_rr']: total_loss += np.array(history['train_loss_rr'])
                    if 'train_loss_pretext' in history and history['train_loss_pretext']: total_loss += np.array(history['train_loss_pretext'])
                    histories[model_name][exp_type]['total'].append(total_loss)

            if best_seed is not None:
                print(f"Melhor modelo para {model_name}/{exp_type} é da Seed {best_seed} (Val Loss: {min_val_loss:.4f})")
                best_models_info[model_name][exp_type] = {'seed': best_seed, 'path': best_run_path}

                # Artefatos a serem copiados
                model_artifact = f"{model_name}_{exp_type}_seed_{best_seed}.pt"
                artifacts_to_copy = [model_artifact, 'test_labels.pt', 'test_scores.pt']

                for artifact in artifacts_to_copy:
                    src = os.path.join(best_run_path, artifact)
                    if os.path.exists(src):
                        # O destino agora é o diretório do modelo dentro de best_models
                        dest = os.path.join(model_output_dir, os.path.basename(src))
                        shutil.copy(src, dest)
                        print(f"  -> Copiado {artifact} para {dest}")
                    else:
                        print(f"  -> Aviso: Artefato {artifact} não encontrado em {best_run_path}")


    return results, histories, best_models_info

def generate_loss_plots(histories, experiment_set_dir):
    """Gera gráficos de comparação para a perda de treinamento total e principal."""
    print("\n--- Gerando Gráficos de Perda ---")
    legend_map = {"baseline": "Baseline", "2_task_multitask": "2-Task Multitask", "3_task_multitask": "3-Task Multitask"}
    loss_types = {"total": "Total Training Loss", "main": "Main Classification Training Loss"}

    for model_name in MODEL_NAMES:
        for loss_key, loss_title in loss_types.items():
            fig, ax = plt.subplots(figsize=(12, 8))
            has_data = False
            for exp_type in legend_map.keys():
                if histories.get(model_name, {}).get(exp_type, {}).get(loss_key):
                    loss_curves = histories[model_name][exp_type][loss_key]
                    if not loss_curves: continue
                    has_data = True
                    max_len = max(len(c) for c in loss_curves)
                    padded_curves = [np.pad(c, (0, max_len - len(c)), 'edge') for c in loss_curves]
                    mean_loss, std_loss = np.mean(padded_curves, axis=0), np.std(padded_curves, axis=0)
                    ax.plot(mean_loss, label=legend_map[exp_type], lw=2.5)
                    ax.fill_between(range(max_len), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)

            if has_data:
                ax.set_title(f'Average {loss_title} for {model_name.upper()}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, which="both", ls="--", c='0.7')
                plot_filepath = os.path.join(experiment_set_dir, f"{model_name}_{loss_key}_loss_comparison.png")
                plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Gráfico de perda salvo em {plot_filepath}")

def plot_roc_curves(best_models_info, experiment_set_dir):
    """Gera um gráfico de curva ROC comparativo para os melhores modelos de cada tipo."""
    print("\n--- Gerando Gráficos ROC/AUC ---")
    best_models_dir = os.path.join(experiment_set_dir, 'best_models')

    for model_name in MODEL_NAMES:
        model_best_dir = os.path.join(best_models_dir, model_name)
        if not os.path.isdir(model_best_dir):
            continue

        fig, ax = plt.subplots(figsize=(10, 10))
        has_data = False

        # Itera sobre os tipos de experimento para plotar cada curva
        for exp_type in ["baseline", "2_task_multitask", "3_task_multitask"]:
            if exp_type in best_models_info.get(model_name, {}):
                info = best_models_info[model_name][exp_type]
                best_seed = info['seed']

                # Caminhos para os artefatos no diretório best_models
                labels_path = os.path.join(model_best_dir, 'test_labels.pt')
                scores_path = os.path.join(model_best_dir, 'test_scores.pt')

                # Verifica se os arquivos de score e label existem para o melhor modelo
                # Nota: Assumimos que os arquivos `test_labels.pt` e `test_scores.pt` corretos
                # foram copiados para o diretório do `exp_type` durante a agregação.
                # Vamos ajustar para procurar o arquivo de score específico do melhor modelo.

                # O ideal é que cada melhor experimento tenha seus próprios scores e labels salvos.
                # A lógica de cópia em `aggregate_and_find_best` precisa garantir isso.
                # Vamos assumir que os arquivos foram copiados e nomeados exclusivamente.

                # Acessando os arquivos que foram copiados para a pasta best_models
                specific_labels_path = os.path.join(info['path'], 'test_labels.pt')
                specific_scores_path = os.path.join(info['path'], 'test_scores.pt')


                if os.path.exists(specific_labels_path) and os.path.exists(specific_scores_path):
                    has_data = True
                    labels = torch.load(specific_labels_path, weights_only=False).numpy()
                    scores = torch.load(specific_scores_path, weights_only=False).numpy()

                    fpr, tpr, _ = roc_curve(labels, scores)
                    auc_score = roc_auc_score(labels, scores)

                    legend_label = f"{exp_type.replace('_', ' ').title()} (Seed: {best_seed}, AUC: {auc_score:.3f})"
                    ax.plot(fpr, tpr, lw=2.5, label=legend_label)
                else:
                    print(f"Aviso: Arquivos de score/label não encontrados para {model_name}/{exp_type} no caminho {info['path']}")


        if has_data:
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve Comparison for Best {model_name.upper()} Models')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.7)

            # Salva o gráfico no diretório principal do experimento
            plot_filepath = os.path.join(experiment_set_dir, f"roc_auc_comparison_{model_name}.png")
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Gráfico ROC/AUC para {model_name} salvo em {plot_filepath}")

def write_summary_file(results, experiment_set_dir):
    """Escreve os resultados agregados finais em um arquivo de texto."""
    print("\n--- Escrevendo Arquivo de Resumo Final ---")
    summary_filepath = os.path.join(experiment_set_dir, "final_results.txt")
    with open(summary_filepath, 'w') as f:
        f.write("="*90 + "\n")
        f.write("           Final Aggregated Results Across All Seeds\n")
        f.write("="*90 + "\n\n")

        for model_name in MODEL_NAMES:
            f.write(f"--- Model: {model_name.upper()} ---\n\n")
            for exp_type in ["baseline", "2_task_multitask", "3_task_multitask"]:
                if not results[model_name][exp_type]: continue

                f.write(f"  Experiment: {exp_type}\n")
                metrics_list = results[model_name][exp_type]

                f.write("    Overall Metrics (Mean ± Std Dev):\n")
                metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'exec_time']
                for key in metric_keys:
                    values = [m[key] for m in metrics_list if key in m]
                    if values:
                        mean, std = np.mean(values), np.std(values)
                        key_name = key.replace('exec_time', 'Execution Time (s)').replace('_', ' ').capitalize()
                        f.write(f"      - {key_name:<20}: {mean:.4f} ± {std:.4f}\n")

                f.write("\n    Data Statistics (from first seed):\n")
                first_run_metrics = metrics_list[0]
                if 'ds1_before' in first_run_metrics:
                    f.write(f"      - Training data (DS1) before downsampling: {first_run_metrics['ds1_before']}\n")
                if 'ds1_after' in first_run_metrics:
                    f.write(f"      - Training data (DS1) after downsampling:  {first_run_metrics['ds1_after']}\n")
                if 'ds1_pretext' in first_run_metrics:
                    f.write(f"      - Training data (DS1) after ECGWavePuzzle: {first_run_metrics['ds1_pretext']}\n")
                if 'ds2_test' in first_run_metrics:
                    f.write(f"      - Test data (DS2) : {first_run_metrics['ds2_test']}\n")
                f.write("-" * 60 + "\n\n")
    print(f"Resumo final dos resultados salvo em {summary_filepath}")

def main():
    """Função principal para executar o pipeline de análise completo."""
    parser = argparse.ArgumentParser(description="Aggregate experiment results.")
    parser.add_argument('experiment_dir', type=str, help='The path to the experiment directory containing the results.')
    args = parser.parse_args()

    experiment_set_dir = args.experiment_dir
    if not os.path.isdir(experiment_set_dir):
        print(f"Error: Directory not found at '{experiment_set_dir}'")
        sys.exit(1)

    print(f"--- Iniciando Agregação para {experiment_set_dir} ---")
    setup_plot_style()

    seeds_filepath = os.path.join(experiment_set_dir, 'seeds.json')
    try:
        with open(seeds_filepath, 'r') as f:
            seeds = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find seeds.json in {experiment_set_dir}. Aborting.")
        return

    results, histories, best_models_info = aggregate_and_find_best(experiment_set_dir, seeds)
    write_summary_file(results, experiment_set_dir)
    generate_loss_plots(histories, experiment_set_dir)
    plot_roc_curves(best_models_info, experiment_set_dir)

    print("\n--- Agregação e Plotagem Concluídas ---")

if __name__ == '__main__':
    main()