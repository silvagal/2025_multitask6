import os
import torch
import random
import json
import time
import subprocess
import sys
import argparse

from main import run_experiment, write_metrics_to_file
from config import (LEARNING_RATE, BATCH_SIZE, EPOCHS, ROOT_OUTPUT_DIR,
                    DATASET, MODEL_NAMES, DEFAULT_SEEDS)

def main():
    """
    Main function to run all experiments for all models and seeds,
    then automatically aggregate the results.
    """
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
                output_dir=exp1_dir
            )
            write_metrics_to_file(test_metrics_1, os.path.join(exp1_dir, "test_metrics.txt"), seed, exp1_str, model_name, LEARNING_RATE, BATCH_SIZE, EPOCHS, exec_time_1, best_val_loss_1, train_stats=train_stats_1, test_stats=test_stats_1, pretext_stats=pretext_stats_1)
            # Renomeia o modelo salvo para incluir detalhes do experimento
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
                output_dir=exp2_dir
            )
            write_metrics_to_file(test_metrics_2, os.path.join(exp2_dir, "test_metrics.txt"), seed, exp2_str, model_name, LEARNING_RATE, BATCH_SIZE, EPOCHS, exec_time_2, best_val_loss_2, train_stats=train_stats_2, test_stats=test_stats_2, pretext_stats=pretext_stats_2)
            model_path = os.path.join(exp2_dir, "best_model.pth")
            if os.path.exists(model_path):
                new_model_path = os.path.join(exp2_dir, f"{model_name}_{exp2_str}_seed_{seed}.pt")
                os.rename(model_path, new_model_path)
                print(f"Modelo renomeado para: {new_model_path}")
            print("="*30 + f" Finished Experiment 2 for {model_name} (Seed: {seed}) " + "="*30 + "\n")

            # --- Experiment 3: Multitask (Arrhythmia + RR Regression + Pretext Task) ---
            if DATASET == 'mit': # Pretext task is only supported for MIT dataset
                exp3_str = "3_task_multitask"
                print("\n" + "="*30 + f" Running Experiment 3: {exp3_str} for {model_name} (Seed: {seed}) " + "="*30)
                exp3_dir = os.path.join(experiment_set_dir, model_name, str(seed), exp3_str)
                _, test_metrics_3, _, _, _, train_stats_3, test_stats_3, pretext_stats_3, exec_time_3, best_val_loss_3 = run_experiment(
                    **common_params,
                    multitask_experiment=True,
                    with_pretext=True,
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


if __name__ == '__main__':
    # This script runs the full suite of experiments as requested and then aggregates the results.
    main()