import datetime
import numpy as np

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
