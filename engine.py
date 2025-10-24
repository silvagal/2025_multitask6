from config import DEVICE
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def train_epoch(model, loader, optimizer, criterion_main, criterion_aux, criterion_pretext,
                multitask_flag, with_pretext_flag, current_epoch, epochs):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {current_epoch + 1:02d}/{epochs} Trn", leave=False)
    total_loss_main, total_loss_rr, total_loss_pretext = 0, 0, 0
    all_preds_main, all_labels_main = [], []

    for batch_idx, batch in enumerate(loop):
        inputs = batch['IEGM_seg'].to(DEVICE)
        labels_main = batch['ac'].to(DEVICE)
        optimizer.zero_grad()

        if not multitask_flag: # Single-task
            preds_main = model(inputs)
            loss_total = criterion_main(preds_main, labels_main)
            total_loss_main += loss_total.item()
            loop.set_postfix(loss=loss_total.item())
        else: # Multitask (2 or 3 tasks)
            labels_rr = batch['rr'].to(DEVICE)

            if with_pretext_flag:
                labels_pretext = batch['pretext'].to(DEVICE)
                preds_main, preds_rr, preds_pretext, log_var_main, log_var_rr, log_var_pretext = model(inputs)

                loss_main_raw = criterion_main(preds_main, labels_main)
                loss_rr_raw = criterion_aux(preds_rr, labels_rr)
                loss_pretext_raw = criterion_pretext(preds_pretext, labels_pretext)

                prec_main = torch.exp(-log_var_main)
                loss_main = prec_main * loss_main_raw + log_var_main

                prec_rr = torch.exp(-log_var_rr)
                loss_rr = prec_rr * loss_rr_raw + log_var_rr

                prec_pretext = torch.exp(-log_var_pretext)
                loss_pretext = prec_pretext * loss_pretext_raw + log_var_pretext

                loss_total = loss_main + loss_rr + loss_pretext

                total_loss_main += loss_main_raw.item()
                total_loss_rr += loss_rr_raw.item()
                total_loss_pretext += loss_pretext_raw.item()

                loop.set_postfix({
                    'Lmain': loss_main_raw.item(), 'Lrr': loss_rr_raw.item(), 'Lpre': loss_pretext_raw.item(),
                    'log_v_main': log_var_main.item(), 'log_v_rr': log_var_rr.item(), 'log_v_pre': log_var_pretext.item()
                })
            else: # 2-task multitask
                preds_main, preds_rr, log_var_main, log_var_rr = model(inputs)

                loss_main_raw = criterion_main(preds_main, labels_main)
                loss_rr_raw = criterion_aux(preds_rr, labels_rr)

                prec_main = torch.exp(-log_var_main)
                loss_main = prec_main * loss_main_raw + log_var_main

                prec_rr = torch.exp(-log_var_rr)
                loss_rr = prec_rr * loss_rr_raw + log_var_rr

                loss_total = loss_main + loss_rr

                total_loss_main += loss_main_raw.item()
                total_loss_rr += loss_rr_raw.item()

                loop.set_postfix({
                    'Lmain': loss_main_raw.item(), 'Lrr': loss_rr_raw.item(),
                    'log_v_main': log_var_main.item(), 'log_v_rr': log_var_rr.item()
                })

        loss_total.backward()
        optimizer.step()

        all_preds_main.extend(preds_main.argmax(dim=1).cpu().numpy())
        all_labels_main.extend(labels_main.cpu().numpy())

    return {
        'loss_main': total_loss_main / len(loader),
        'loss_rr': total_loss_rr / len(loader),
        'loss_pretext': total_loss_pretext / len(loader),
        'accuracy': accuracy_score(all_labels_main, all_preds_main)
    }

def evaluate_epoch(model, loader, criterion_main, criterion_aux, criterion_pretext,
                   multitask_flag, with_pretext_flag, metrics=False):
    model.eval()
    total_loss_main, total_loss_rr, total_loss_pretext = 0.0, 0.0, 0.0
    all_preds_main, all_labels_main, all_scores_main = [], [], []

    with torch.inference_mode():
        for batch in loader:
            inputs = batch['IEGM_seg'].to(DEVICE)
            labels_main = batch['ac'].to(DEVICE)
            loss_main, loss_rr, loss_pretext = 0, 0, 0

            if not multitask_flag: # Single-task
                preds_main = model(inputs)
                loss_main = criterion_main(preds_main, labels_main)
            else: # Multitask (2 or 3 tasks)
                labels_rr = batch['rr'].to(DEVICE)
                if with_pretext_flag:
                    preds_main, preds_rr, preds_pretext, _, _, _ = model(inputs)
                    if 'pretext' in batch:
                        labels_pretext = batch['pretext'].to(DEVICE)
                        loss_pretext = criterion_pretext(preds_pretext, labels_pretext)
                        total_loss_pretext += loss_pretext.item()
                else:  # 2-task multitask
                    preds_main, preds_rr, _, _ = model(inputs)

                loss_main = criterion_main(preds_main, labels_main)
                loss_rr = criterion_aux(preds_rr, labels_rr)

            total_loss_main += loss_main.item()
            if isinstance(loss_rr, torch.Tensor): total_loss_rr += loss_rr.item()

            if metrics:
                all_labels_main.extend(labels_main.cpu().numpy())
                all_preds_main.extend(preds_main.argmax(dim=1).cpu().numpy())
                # Adiciona a probabilidade softmax para a classe positiva (classe 1)
                all_scores_main.extend(torch.softmax(preds_main, dim=1)[:, 1].cpu().numpy())

    avg_loss_main = total_loss_main / len(loader)
    avg_loss_rr = total_loss_rr / len(loader) if multitask_flag else 0
    avg_loss_pretext = total_loss_pretext / len(loader) if with_pretext_flag else 0

    if not metrics:
        return {'loss_main': avg_loss_main, 'loss_rr': avg_loss_rr, 'loss_pretext': avg_loss_pretext}, None, None, None, None, None

    # --- Metrics Calculation ---
    if len(all_labels_main) > 0 and len(all_preds_main) == len(all_labels_main):
        labels = [0, 1]
        acc = accuracy_score(all_labels_main, all_preds_main)
        prec = precision_score(all_labels_main, all_preds_main, average='weighted', zero_division=0, labels=labels)
        rec = recall_score(all_labels_main, all_preds_main, average='weighted', zero_division=0, labels=labels)

        class_names = [str(l) for l in labels]
        cm = confusion_matrix(all_labels_main, all_preds_main, labels=labels)
        report = classification_report(all_labels_main, all_preds_main, target_names=class_names, zero_division=0, labels=labels)

        spec = 0.0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics_dict = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'specificity': spec,
            'loss_main': avg_loss_main, 'loss_rr': avg_loss_rr, 'loss_pretext': avg_loss_pretext
        }
        return metrics_dict, cm, report, all_labels_main, all_preds_main, all_scores_main
    else:
        print("Warning: No labels/preds for metric calculation in evaluate_epoch.")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'specificity': 0,
                'loss_main': avg_loss_main, 'loss_rr': avg_loss_rr, 'loss_pretext': avg_loss_pretext}, np.array([]), "", [], [], []