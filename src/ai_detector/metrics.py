import numpy
from typing import Any
from sklearn.metrics import ( 
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_curve, 
    roc_auc_score,
    auc,
    precision_recall_curve, 
    confusion_matrix,
)

def get_roc_optimal_metrics(real_labels, predicted_probs):

    fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    optimal_idx = numpy.argmax(tpr - fpr)
    # optimal threshold is the threshold that maximizes the TPR - FPR
    optimal_threshold = thresholds[optimal_idx]

    predictions = [1 if prob >= optimal_threshold else 0 for prob in predicted_probs]
    conf_matrix = confusion_matrix(real_labels, predictions)
    precision = precision_score(real_labels, predictions)
    recall = recall_score(real_labels, predictions)
    f1 = f1_score(real_labels, predictions)
    accuracy = accuracy_score(real_labels, predictions)
    # value corresponding to an FPR of 0.0001 (0.01%) from the ROC curve
    tpr_at_fpr_0_01 = numpy.interp(0.01/100.0, fpr, tpr) 
    threshold_at_fpr_0_01 = numpy.interp(0.01/100.0, fpr, thresholds)

    return {
        "auc_roc": round(float(roc_auc), 4), 
        "optimal_threshold": round(float(optimal_threshold), 4), 
        "conf_matrix": conf_matrix.tolist(), 
        "precision": round(float(precision), 4), 
        "recall": round(float(recall), 4), 
        "f1": round(float(f1), 4), 
        "accuracy": round(float(accuracy), 4), 
        "tpr_at_fpr_0_01": round(float(tpr_at_fpr_0_01), 4),
        "threshold_at_fpr_0_01": round(float(threshold_at_fpr_0_01), 4)
    }


def get_optimal_threshold(real_labels, predicted_probs, mode: str = "accuracy"):
    thresholds = numpy.linspace(0.0, 1.0, 100)
    metrics = {}
    for threshold in thresholds:
        predictions = [1 if prob >= threshold else 0 for prob in predicted_probs]
        if mode == "accuracy":
            metrics[threshold] = accuracy_score(real_labels, predictions)
        elif mode == "f1":
            metrics[threshold] = f1_score(real_labels, predictions)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    return max(metrics, key=metrics.get)

def get_precision_recall_curve(real_labels, predicted_probs):
    precision, recall, _ = precision_recall_curve(real_labels, predicted_probs)
    pr_auc = auc(recall, precision)
    auc_roc = roc_auc_score(real_labels, predicted_probs)
    return {
        "precision": precision.tolist(), 
        "recall": recall.tolist(), 
        "pr_auc": round(float(pr_auc), 4),
        "auc_roc": round(float(auc_roc), 4)
    }

def get_metrics(real_labels, predictions, positive_label: Any = 1):
    precision = precision_score(real_labels, predictions, pos_label=positive_label)
    recall = recall_score(real_labels, predictions, pos_label=positive_label)
    f1 = f1_score(real_labels, predictions, pos_label=positive_label)
    accuracy = accuracy_score(real_labels, predictions)

    return {
        "precision": round(float(precision), 4), 
        "recall": round(float(recall), 4), 
        "f1": round(float(f1), 4), 
        "accuracy": round(float(accuracy), 4),
    }
