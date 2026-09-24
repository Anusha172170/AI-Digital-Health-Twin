"""Binary metrics at an explicitly recorded decision threshold."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)


def evaluate(y, probabilities, threshold):
    pred = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'threshold': float(threshold), 'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'specificity': float(tn / (tn + fp)), 'f1': f1_score(y, pred),
        'roc_auc': roc_auc_score(y, probabilities),
        'average_precision': average_precision_score(y, probabilities),
        'brier_score': brier_score_loss(y, probabilities),
        'log_loss': log_loss(y, probabilities),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}
