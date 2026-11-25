import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from pathlib import Path

CONFUSION_MATRIX_PATH = Path(__file__).parent.parent / "reports" / "confusion_matrix.png"

def classification_report_df(report):
    """Convert sklearn classification report to DataFrame."""
    if isinstance(report, str):
        report = classification_report(y_true=[], y_pred=[], output_dict=True)
    
    df = pd.DataFrame().from_dict(report).transpose()
    return df.round(4)

def plot_confusion_matrix(y_true, y_pred, classes=['Negative', 'Positive']):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    
    return cm

def evaluate_model(y_true, y_pred, target_names=['Negative', 'Positive']):
    """Comprehensive model evaluation."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    metrics = {
        'accuracy': accuracy,
        'precision_macro': p_macro,
        'recall_macro': r_macro,
        'f1_macro': f_macro
    }
    
    # Per-class metrics
    class_metrics = {}
    for i, cls in enumerate(target_names):
        class_metrics[cls] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    return {
        'overall': metrics,
        'per_class': class_metrics,
        'report': classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    }

def save_results(results, filepath):
    """Save evaluation results to JSON/CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to save-friendly format
    save_data = {
        'accuracy': results['overall']['accuracy'],
        'per_class': {k: v for k, v in results['per_class'].items()}
    }
    
    import json
    with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {filepath.with_suffix('.json')}")