from config import DEVICE
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

def train_epoch(model, loader, optimizer, criterion_main, criterion_aux, criterion_pretext,
                multitask_flag, with_pretext_flag, with_rr_flag, current_epoch, epochs):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {current_epoch + 1:02d}/{epochs} Trn", leave=False)
    total_loss_main, total_loss_rr, total_loss_pretext = 0, 0, 0
    all_preds_main, all_labels_main = [], []

    for batch_idx, batch in enumerate(loop):
        inputs = batch['IEGM_seg'].to(DEVICE)
        labels_main = batch['ac'].to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)

        if not multitask_flag:
            preds_main = outputs
            loss_total = criterion_main(preds_main, labels_main)
            total_loss_main += loss_total.item()
            loop.set_postfix(loss=loss_total.item())
        else:
            # Unpack outputs and log_vars based on the model's configuration
            num_tasks = 1 + with_rr_flag + with_pretext_flag
            preds = outputs[:num_tasks]
            log_vars = outputs[num_tasks:]

            preds_main = preds[0]
            log_var_main = log_vars[0]

            loss_main_raw = criterion_main(preds_main, labels_main)
            prec_main = torch.exp(-log_var_main)
            loss_total = prec_main * loss_main_raw + log_var_main
            total_loss_main += loss_main_raw.item()

            postfix_dict = {'Lmain': loss_main_raw.item(), 'log_v_main': log_var_main.item()}

            task_idx = 1
            if with_rr_flag:
                preds_rr = preds[task_idx]
                labels_rr = batch['rr'].to(DEVICE)
                log_var_rr = log_vars[task_idx]

                loss_rr_raw = criterion_aux(preds_rr, labels_rr)
                prec_rr = torch.exp(-log_var_rr)
                loss_total += prec_rr * loss_rr_raw + log_var_rr
                total_loss_rr += loss_rr_raw.item()

                postfix_dict['Lrr'] = loss_rr_raw.item()
                postfix_dict['log_v_rr'] = log_var_rr.item()
                task_idx += 1

            if with_pretext_flag:
                preds_pretext = preds[task_idx]
                labels_pretext = batch['pretext'].to(DEVICE)
                log_var_pretext = log_vars[task_idx]

                loss_pretext_raw = criterion_pretext(preds_pretext, labels_pretext)
                prec_pretext = torch.exp(-log_var_pretext)
                loss_total += prec_pretext * loss_pretext_raw + log_var_pretext
                total_loss_pretext += loss_pretext_raw.item()

                postfix_dict['Lpre'] = loss_pretext_raw.item()
                postfix_dict['log_v_pre'] = log_var_pretext.item()

            loop.set_postfix(postfix_dict)

        loss_total.backward()
        optimizer.step()

        all_preds_main.extend(preds_main.argmax(dim=1).cpu().numpy())
        all_labels_main.extend(labels_main.cpu().numpy())

    return {
        'loss_main': total_loss_main / len(loader),
        'loss_rr': total_loss_rr / len(loader) if with_rr_flag else 0,
        'loss_pretext': total_loss_pretext / len(loader) if with_pretext_flag else 0,
        'accuracy': accuracy_score(all_labels_main, all_preds_main)
    }

def evaluate_epoch(model, loader, criterion_main, criterion_aux, criterion_pretext,
                   multitask_flag, with_pretext_flag, with_rr_flag, metrics=False):
    model.eval()
    total_loss_main, total_loss_rr, total_loss_pretext = 0.0, 0.0, 0.0
    all_preds_main, all_labels_main, all_scores_main = [], [], []

    with torch.inference_mode():
        for batch in loader:
            inputs = batch['IEGM_seg'].to(DEVICE)
            labels_main = batch['ac'].to(DEVICE)

            outputs = model(inputs)

            if not multitask_flag:
                preds_main = outputs
                loss_main = criterion_main(preds_main, labels_main)
                total_loss_main += loss_main.item()
            else:
                num_tasks = 1 + with_rr_flag + with_pretext_flag
                preds = outputs[:num_tasks]

                preds_main = preds[0]
                loss_main = criterion_main(preds_main, labels_main)
                total_loss_main += loss_main.item()

                task_idx = 1
                if with_rr_flag:
                    preds_rr = preds[task_idx]
                    labels_rr = batch['rr'].to(DEVICE)
                    loss_rr = criterion_aux(preds_rr, labels_rr)
                    total_loss_rr += loss_rr.item()
                    task_idx += 1

                if with_pretext_flag and 'pretext' in batch:
                    preds_pretext = preds[task_idx]
                    labels_pretext = batch['pretext'].to(DEVICE)
                    loss_pretext = criterion_pretext(preds_pretext, labels_pretext)
                    total_loss_pretext += loss_pretext.item()

            if metrics:
                all_labels_main.extend(labels_main.cpu().numpy())
                all_preds_main.extend(preds_main.argmax(dim=1).cpu().numpy())
                all_scores_main.extend(torch.softmax(preds_main, dim=1)[:, 1].cpu().numpy())

    avg_loss_main = total_loss_main / len(loader)
    avg_loss_rr = total_loss_rr / len(loader) if with_rr_flag else 0
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