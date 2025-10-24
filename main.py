import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import random
import numpy as np
import datetime

from config import (set_seed, DEVICE, WEIGHT_DECAY, SAMPLING_RATE, PATIENCE,
                    EXPERIMENT_TYPE, LEARNING_RATE, BATCH_SIZE, EPOCHS,
                    BASE_PATH, NUM_WORKERS, PERSISTENT_WORKERS, DATASET, ROOT_OUTPUT_DIR, PROJECT_ROOT,
                    QRS_COMPLEX_PROPORTION, DEFAULT_SEEDS)
from utils.augmentation import augment_signal
from utils.iegmDataClass import SimpleDataset, normalize_signal
from utils.rr_computing import Scaler
from models.hydranet import HydraNet
from models.vanet import VANetFeatureExtractor, VANetOriginal
from models.lightnet import LightNetFeatureExtractor, LightNet
from models.heavynet import HeavyNetFeatureExtractor, HeavyNet
from utils.rr_computing import compute_rr_features
from utils.pretext_task import generate_pretext_sequences
from engine import train_epoch, evaluate_epoch
from utils.plot_output_results import plot_confusion_matrix, plot_training_history
from utils.plot_signals_utils import plot_signals_with_r_peaks, plot_signals_with_annotations
from database.mit.process_mit import main as process_mit_data
from utils.file_utils import write_metrics_to_file

import time

def run_experiment(seed, multitask_experiment, with_pretext, with_rr, learning_rate, batch_size, epochs, output_dir, model_name="vanet"):
    start_time = time.time()
    set_seed(seed)
    experiment_str = f"Model: {model_name}, Seed: {seed}, Multitask: {multitask_experiment}, Pretext: {with_pretext}, RR: {with_rr}, Device: {DEVICE}, Dataset: {DATASET}"
    print(f"Running experiment with {experiment_str}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Data Loading & Processing ---
    if DATASET == 'mit':
        mit_data_path = os.path.join(BASE_PATH, 'mit')
        train_signals_path = os.path.join(mit_data_path, "mit_train_signals.pt")
        if not os.path.exists(train_signals_path):
            print("Processed MIT data not found. Generating now...")
            process_mit_data()
            print("MIT data generation complete.")

        # Carrega os dados de treino pré-divididos
        processed_train_signals_list = torch.load(train_signals_path, weights_only=False)
        train_labels = torch.load(os.path.join(mit_data_path, "mit_train_labels.pt"), weights_only=False)
        train_r_peaks = torch.load(os.path.join(mit_data_path, "mit_train_r_peaks.pt"), weights_only=False)

        # Carrega os dados de teste pré-divididos
        processed_test_signals_list = torch.load(os.path.join(mit_data_path, "mit_test_signals.pt"), weights_only=False)
        test_labels = torch.load(os.path.join(mit_data_path, "mit_test_labels.pt"), weights_only=False)
        test_r_peaks = torch.load(os.path.join(mit_data_path, "mit_test_r_peaks.pt"), weights_only=False)

        # Carrega as features RR se for um experimento multitask
        rr_train, rr_test = None, None
        if multitask_experiment:
            rr_train = torch.load(os.path.join(mit_data_path, "mit_train_rr_norm.pt"), weights_only=False)
            rr_test = torch.load(os.path.join(mit_data_path, "mit_test_rr_norm.pt"), weights_only=False)
    else:
        raise NotImplementedError("A lógica de carregamento para datasets diferentes de 'mit' não está implementada.")

    # --- Downsampling logic for the training set ---
    train_stats = {}
    labels_np = train_labels.numpy()
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    class_counts_before = {int(k): int(v) for k, v in zip(unique_labels, counts)}
    train_stats['before'] = class_counts_before
    print(f"DS1 (Train) Original distribution: {class_counts_before}")

    class_0_indices = np.where(labels_np == 0)[0]
    class_1_indices = np.where(labels_np == 1)[0]

    if len(class_0_indices) > 0 and len(class_1_indices) > 0:
        if len(class_0_indices) > len(class_1_indices):
            majority_indices, minority_indices = class_0_indices, class_1_indices
        else:
            majority_indices, minority_indices = class_1_indices, class_0_indices

        downsampled_majority_indices = np.random.choice(majority_indices, size=len(minority_indices), replace=False)
        balanced_indices = np.concatenate([downsampled_majority_indices, minority_indices])
        np.random.shuffle(balanced_indices)

        processed_train_signals_list = [processed_train_signals_list[i] for i in balanced_indices]
        train_labels = train_labels[balanced_indices]
        if DATASET == 'mit':
            train_r_peaks = [train_r_peaks[i] for i in balanced_indices]
        if multitask_experiment:
            rr_train = rr_train[balanced_indices]

        unique_labels_after, counts_after = np.unique(train_labels.numpy(), return_counts=True)
        class_counts_after = {int(k): int(v) for k, v in zip(unique_labels_after, counts_after)}
        train_stats['after'] = class_counts_after
        print(f"DS1 (Train) Downsampled distribution: {class_counts_after}")
    else:
        print("Warning: Training set contains only one class. Skipping downsampling.")
        train_stats['after'] = class_counts_before

    # --- Pretext Task Data Augmentation ---
    pretext_train_labels = None
    pretext_stats = {} # Initialize pretext_stats
    if with_pretext:
        if DATASET != 'mit':
            raise ValueError("Pretext task is currently only supported for the MIT dataset.")
        if not multitask_experiment:
            raise ValueError("Pretext task must be run with multitask_experiment=True.")

        print("Applying pretext task data augmentation...")
        # Capture stats before
        labels_before_pretext = train_labels.numpy()
        unique_before, counts_before = np.unique(labels_before_pretext, return_counts=True)
        pretext_stats['before'] = {int(k): int(v) for k, v in zip(unique_before, counts_before)}

        processed_train_signals_list, train_labels, pretext_train_labels, rr_train = generate_pretext_sequences(
            signals=torch.stack(processed_train_signals_list) if isinstance(processed_train_signals_list, list) else processed_train_signals_list,
            r_peaks=train_r_peaks,
            main_labels=train_labels,
            rr_features=rr_train,
            qrs_proportion=QRS_COMPLEX_PROPORTION
        )

        # Capture stats after
        labels_after_pretext = train_labels.numpy()
        unique_after, counts_after = np.unique(labels_after_pretext, return_counts=True)
        pretext_stats['after'] = {int(k): int(v) for k, v in zip(unique_after, counts_after)}

        print(f"Training set size after pretext augmentation: {len(processed_train_signals_list)}")

    # --- Dataset and DataLoader ---
    full_train_dataset = SimpleDataset(processed_train_signals_list, train_labels, rr_train, pretext_train_labels)
    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = SimpleDataset(processed_test_signals_list, test_labels, rr_test, None)

    persistent_workers = PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=persistent_workers)

    # --- Model, Optimizer, Criterion ---
    if model_name == "vanet":
        feature_extractor = VANetFeatureExtractor()
        base_model = VANetOriginal
    elif model_name == "lightnet":
        feature_extractor = LightNetFeatureExtractor()
        base_model = LightNet
    elif model_name == "heavynet":
        feature_extractor = HeavyNetFeatureExtractor()
        base_model = HeavyNet
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if multitask_experiment:
        model = HydraNet(feature_extractor, with_pretext_task=with_pretext, with_rr_task=with_rr).to(DEVICE)
    else:
        model = base_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    criterion_main = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()
    criterion_pretext = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE // 2)

    # --- Training Loop ---
    best_val_loss = float('inf')
    best_model_state = None
    history = {
        'train_loss_main': [], 'train_loss_rr': [], 'train_loss_pretext': [], 'train_acc': [],
        'val_loss_main': [], 'val_loss_rr': [], 'val_loss_pretext': [], 'val_acc': []
    }
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_main, criterion_aux, criterion_pretext,
            multitask_flag=multitask_experiment, with_pretext_flag=with_pretext, with_rr_flag=with_rr,
            current_epoch=epoch, epochs=epochs
        )
        val_metrics, _, _, _, _, _ = evaluate_epoch(
            model, val_loader, criterion_main, criterion_aux, criterion_pretext,
            multitask_flag=multitask_experiment, with_pretext_flag=with_pretext, with_rr_flag=with_rr, metrics=True
        )

        history['train_loss_main'].append(train_metrics['loss_main'])
        history['train_loss_rr'].append(train_metrics['loss_rr'])
        history['train_loss_pretext'].append(train_metrics['loss_pretext'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss_main'].append(val_metrics['loss_main'])
        history['val_loss_rr'].append(val_metrics['loss_rr'])
        history['val_loss_pretext'].append(val_metrics['loss_pretext'])
        history['val_acc'].append(val_metrics['accuracy'])

        train_loss_str = f"Trn L(M/R/P): {train_metrics['loss_main']:.4f}/{train_metrics['loss_rr']:.4f}/{train_metrics['loss_pretext']:.4f}"
        val_loss_str = f"Val L(M/R/P): {val_metrics['loss_main']:.4f}/{val_metrics['loss_rr']:.4f}/{val_metrics['loss_pretext']:.4f}"
        print(f"E{epoch+1:02d}: {train_loss_str}, Trn Acc: {train_metrics['accuracy']:.4f} | {val_loss_str}, Val Acc: {val_metrics['accuracy']:.4f}")

        scheduler.step(val_metrics['loss_main'])
        if val_metrics['loss_main'] < best_val_loss:
            best_val_loss = val_metrics['loss_main']
            best_model_state = model.state_dict()

    # --- Final Evaluation ---
    if best_model_state:
        model.load_state_dict(best_model_state)
        # Salva o melhor modelo
        model_save_path = os.path.join(output_dir, "best_model.pth")
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to {model_save_path}")


    test_metrics, cm, report, test_labels_list, _, test_scores = evaluate_epoch(
        model, test_loader, criterion_main, criterion_aux, criterion_pretext,
        multitask_flag=multitask_experiment, with_pretext_flag=with_pretext, with_rr_flag=with_rr, metrics=True
    )
    test_metrics['report'] = report
    test_metrics['cm'] = cm

    # Salva os rótulos e pontuações de teste para a análise ROC
    labels_save_path = os.path.join(output_dir, "test_labels.pt")
    scores_save_path = os.path.join(output_dir, "test_scores.pt")
    torch.save(torch.tensor(test_labels_list), labels_save_path)
    torch.save(torch.tensor(test_scores), scores_save_path)
    print(f"Test labels and scores saved for ROC analysis.")

    # --- Artifact Plotting and History Saving ---
    if DATASET == 'mit':
        class_names = ['0', '1']
    else:
        class_names = [str(i) for i in range(len(np.unique(raw_test_labels.numpy())))]

    # Plotting
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, "confusion_matrix.png"))
    plot_training_history(history, output_dir)

    # Save history object for later aggregation
    history_save_path = os.path.join(output_dir, "history.pt")
    torch.save(history, history_save_path)
    print(f"Training history saved to {history_save_path}")

    # --- Test Data Statistics ---
    test_labels_np = test_labels.numpy()
    unique_labels_test, counts_test = np.unique(test_labels_np, return_counts=True)
    test_class_counts = {int(k): int(v) for k, v in zip(unique_labels_test, counts_test)}
    print(f"DS2 (Test) distribution: {test_class_counts}")

    r_peaks_to_return = test_r_peaks if DATASET == 'mit' else None

    end_time = time.time()
    execution_time = end_time - start_time

    return history, test_metrics, processed_test_signals_list, test_labels, r_peaks_to_return, train_stats, test_class_counts, pretext_stats, execution_time, best_val_loss


if __name__ == '__main__':
    import json
    import time
    import subprocess
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run a set of multitask learning experiments.")
    parser.add_argument('--seeds', type=str, help='Comma-separated list of seeds to use (e.g., "10,20,30").')
    args = parser.parse_args()

    if args.seeds:
        try:
            seeds = [int(seed.strip()) for seed in args.seeds.split(',')]
        except ValueError:
            print("Error: Invalid seeds format. Please provide a comma-separated list of integers.")
            sys.exit(1)
    else:
        seeds = DEFAULT_SEEDS

    # Create a unique directory for this set of experiments based on the seeds
    seeds_str = "_".join(map(str, seeds))
    experiment_set_dir = os.path.join(ROOT_OUTPUT_DIR, f"experiment_seeds_{seeds_str}")
    os.makedirs(experiment_set_dir, exist_ok=True)
    print(f"Starting experiment set. Results will be saved in: {experiment_set_dir}")

    # Save the seeds used for this run for the aggregation script
    seeds_filepath = os.path.join(experiment_set_dir, 'seeds.json')
    with open(seeds_filepath, 'w') as f:
        json.dump(seeds, f)
    print(f"Using seeds: {seeds}")

    for model_name in MODEL_NAMES:
        for seed in seeds:
            common_params = {
                "seed": seed,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "model_name": model_name,
            }

            # --- Experiment 1: Baseline (Arrhythmia Classification Only) ---
            exp1_str = "baseline"
            print("\n" + "="*30 + f" Running Experiment 1: {exp1_str} for {model_name} (Seed: {seed}) " + "="*30)
            exp1_dir = os.path.join(experiment_set_dir, model_name, str(seed), exp1_str)
            _, test_metrics_1, _, _, _, train_stats_1, test_stats_1, pretext_stats_1, exec_time_1, best_val_loss_1 = run_experiment(
                **common_params,
                multitask_experiment=False,
                with_pretext=False,
                with_rr=False,
                output_dir=exp1_dir
            )
            write_metrics_to_file(test_metrics_1, os.path.join(exp1_dir, "test_metrics.txt"), seed, exp1_str, model_name, LEARNING_RATE, BATCH_SIZE, EPOCHS, exec_time_1, best_val_loss_1, train_stats=train_stats_1, test_stats=test_stats_1, pretext_stats=pretext_stats_1)
            model_path = os.path.join(exp1_dir, "best_model.pth")
            if os.path.exists(model_path):
                new_model_path = os.path.join(exp1_dir, f"{model_name}_{exp1_str}_seed_{seed}.pt")
                os.rename(model_path, new_model_path)
                print(f"Modelo renomeado para: {new_model_path}")
            print("="*30 + f" Finished Experiment 1 for {model_name} (Seed: {seed}) " + "="*30 + "\n")

            # --- Experiment 2: Multitask (Arrhythmia Classification + RR Regression) ---
            exp2_str = "2_task_multitask"
            print("\n" + "="*30 + f" Running Experiment 2: {exp2_str} for {model_name} (Seed: {seed}) " + "="*30)
            exp2_dir = os.path.join(experiment_set_dir, model_name, str(seed), exp2_str)
            _, test_metrics_2, _, _, _, train_stats_2, test_stats_2, pretext_stats_2, exec_time_2, best_val_loss_2 = run_experiment(
                **common_params,
                multitask_experiment=True,
                with_pretext=False,
                with_rr=True,
                output_dir=exp2_dir
            )
            write_metrics_to_file(test_metrics_2, os.path.join(exp2_dir, "test_metrics.txt"), seed, exp2_str, model_name, LEARNING_RATE, BATCH_SIZE, EPOCHS, exec_time_2, best_val_loss_2, train_stats=train_stats_2, test_stats=test_stats_2, pretext_stats=pretext_stats_2)
            model_path = os.path.join(exp2_dir, "best_model.pth")
            if os.path.exists(model_path):
                new_model_path = os.path.join(exp2_dir, f"{model_name}_{exp2_str}_seed_{seed}.pt")
                os.rename(model_path, new_model_path)
                print(f"Modelo renomeado para: {new_model_path}")
            print("="*30 + f" Finished Experiment 2 for {model_name} (Seed: {seed}) " + "="*30 + "\n")

            # --- Experiment 3: Multitask (Arrhythmia Classification + Pretext Task) ---
            if DATASET == 'mit': # Pretext task is only supported for MIT dataset
                exp3_str = "2_task_pretext"
                print("\n" + "="*30 + f" Running Experiment 3: {exp3_str} for {model_name} (Seed: {seed}) " + "="*30)
                exp3_dir = os.path.join(experiment_set_dir, model_name, str(seed), exp3_str)
                _, test_metrics_3, _, _, _, train_stats_3, test_stats_3, pretext_stats_3, exec_time_3, best_val_loss_3 = run_experiment(
                    **common_params,
                    multitask_experiment=True,
                    with_pretext=True,
                    with_rr=False,
                    output_dir=exp3_dir
                )
                write_metrics_to_file(test_metrics_3, os.path.join(exp3_dir, "test_metrics.txt"), seed, exp3_str, model_name, LEARNING_RATE, BATCH_SIZE, EPOCHS, exec_time_3, best_val_loss_3, train_stats=train_stats_3, test_stats=test_stats_3, pretext_stats=pretext_stats_3)
                model_path = os.path.join(exp3_dir, "best_model.pth")
                if os.path.exists(model_path):
                    new_model_path = os.path.join(exp3_dir, f"{model_name}_{exp3_str}_seed_{seed}.pt")
                    os.rename(model_path, new_model_path)
                    print(f"Modelo renomeado para: {new_model_path}")
                print("="*30 + f" Finished Experiment 3 for {model_name} (Seed: {seed}) " + "="*30 + "\n")
            else:
                print(f"Skipping 2-task pretext experiment for {model_name} (Seed: {seed}) as dataset is not 'mit'.")

            # --- Experiment 4: Multitask (Arrhythmia + RR Regression + Pretext Task) ---
            if DATASET == 'mit': # Pretext task is only supported for MIT dataset
                exp4_str = "3_task_multitask"
                print("\n" + "="*30 + f" Running Experiment 4: {exp4_str} for {model_name} (Seed: {seed}) " + "="*30)
                exp4_dir = os.path.join(experiment_set_dir, model_name, str(seed), exp4_str)
                _, test_metrics_4, _, _, _, train_stats_4, test_stats_4, pretext_stats_4, exec_time_4, best_val_loss_4 = run_experiment(
                    **common_params,
                    multitask_experiment=True,
                    with_pretext=True,
                    with_rr=True,
                    output_dir=exp4_dir
                )
                write_metrics_to_file(test_metrics_4, os.path.join(exp4_dir, "test_metrics.txt"), seed, exp4_str, model_name, LEARNING_RATE, BATCH_SIZE, EPOCHS, exec_time_4, best_val_loss_4, train_stats=train_stats_4, test_stats=test_stats_4, pretext_stats=pretext_stats_4)
                model_path = os.path.join(exp4_dir, "best_model.pth")
                if os.path.exists(model_path):
                    new_model_path = os.path.join(exp4_dir, f"{model_name}_{exp4_str}_seed_{seed}.pt")
                    os.rename(model_path, new_model_path)
                    print(f"Modelo renomeado para: {new_model_path}")
                print("="*30 + f" Finished Experiment 4 for {model_name} (Seed: {seed}) " + "="*30 + "\n")
            else:
                print(f"Skipping 3-task experiment for {model_name} (Seed: {seed}) as dataset is not 'mit'.")

    # --- Automatically run the aggregation script ---
    print("\n" + "="*30 + " All experiments complete. Running aggregation. " + "="*30)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        aggregation_script_path = os.path.join(script_dir, "aggregate_results.py")

        # Pass the experiment directory as an argument to the aggregation script
        subprocess.run(["python", aggregation_script_path, experiment_set_dir], check=True)
        print("="*30 + " Aggregation complete. " + "="*30 + "\n")
    except FileNotFoundError:
        print("\nError: 'aggregate_results.py' not found. Skipping aggregation.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during aggregation: {e}")