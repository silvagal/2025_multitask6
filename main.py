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
from utils.plot import plot_confusion_matrix, plot_training_history, plot_signals_with_r_peaks, plot_signals_with_annotations
from database.mit.process_mit import main as process_mit_data

def write_metrics_to_file(metrics, output_path, seed, experiment_type_str, model_name, lr, batch_size, epochs, exec_time, best_val_loss, train_stats=None, test_stats=None, pretext_stats=None):
    """Saves a comprehensive dictionary of metrics and experiment parameters to a text file."""
    with open(output_path, 'w') as f:
        f.write("="*30 + " Experiment Details " + "="*30 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Experiment Type: {experiment_type_str}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Execution Time (s): {exec_time:.2f}\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write("="*30 + "====================" + "="*30 + "\n\n")

        if train_stats:
            f.write("--- Data Statistics ---\n")
            if 'before' in train_stats:
                f.write(f"DS1_before_downsampling: {train_stats['before']}\n")
            if 'after' in train_stats:
                f.write(f"DS1_after_downsampling: {train_stats['after']}\n")

        if pretext_stats and pretext_stats.get('after'):
            # This captures the state *after* the ECGWavePuzzle augmentation
            f.write(f"DS1_after_ECGWavePuzzle: {pretext_stats['after']}\n")

        if test_stats:
            f.write(f"DS2_test_data: {test_stats}\n")
            total_test_samples = sum(test_stats.values())
            f.write(f"Total_Test_Samples: {total_test_samples}\n")
        f.write("\n")

        f.write("--- Scalar Metrics ---\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"Specificity: {metrics['specificity']:.4f}\n")
        f.write(f"Main Task Loss: {metrics['loss_main']:.4f}\n")
        if 'loss_rr' in metrics and metrics['loss_rr'] > 0:
            f.write(f"Auxiliary Task Loss: {metrics['loss_rr']:.4f}\n")
        if 'loss_pretext' in metrics and metrics['loss_pretext'] > 0:
            f.write(f"Pretext Task Loss: {metrics['loss_pretext']:.4f}\n")

        f.write("\n--- Classification Report ---\n")
        f.write(metrics['report'])

        f.write("\n--- Confusion Matrix ---\n")
        f.write(np.array2string(metrics['cm']))
    print(f"Test metrics and data stats saved to {output_path}")

import time

def run_experiment(seed, multitask_experiment, with_pretext, learning_rate, batch_size, epochs, output_dir, model_name="vanet"):
    start_time = time.time()
    set_seed(seed)
    experiment_str = f"Model: {model_name}, Seed: {seed}, Multitask: {multitask_experiment}, Pretext: {with_pretext}, Device: {DEVICE}, Dataset: {DATASET}"
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
        model = HydraNet(feature_extractor, with_pretext_task=with_pretext).to(DEVICE)
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
            multitask_flag=multitask_experiment, with_pretext_flag=with_pretext,
            current_epoch=epoch, epochs=epochs
        )
        val_metrics, _, _, _, _, _ = evaluate_epoch(
            model, val_loader, criterion_main, criterion_aux, criterion_pretext,
            multitask_flag=multitask_experiment, with_pretext_flag=with_pretext, metrics=True
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
        multitask_flag=multitask_experiment, with_pretext_flag=with_pretext, metrics=True
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