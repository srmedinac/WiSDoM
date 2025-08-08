import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_predictions(y_true, y_pred, variance=None, threshold=0.05):
    """
    Evaluate model predictions with optional variance filtering
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        variance: Prediction variance (optional)
        threshold: Variance threshold for filtering
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Convert continuous to discrete if needed
    if y_pred.max() <= 1.0:
        y_pred_discrete = np.round(y_pred * 5)
        y_true_discrete = np.round(y_true * 5)
    else:
        y_pred_discrete = y_pred
        y_true_discrete = y_true
    
    # Basic metrics
    metrics['kappa'] = cohen_kappa_score(y_true_discrete, y_pred_discrete, weights='quadratic')
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['accuracy'] = np.mean(y_true_discrete == y_pred_discrete)
    
    # Metrics with variance filtering
    if variance is not None:
        confident_mask = variance < threshold
        if confident_mask.sum() > 0:
            metrics['confident_kappa'] = cohen_kappa_score(
                y_true_discrete[confident_mask],
                y_pred_discrete[confident_mask],
                weights='quadratic'
            )
            metrics['confident_mae'] = mean_absolute_error(
                y_true[confident_mask],
                y_pred[confident_mask]
            )
            metrics['confident_accuracy'] = np.mean(
                y_true_discrete[confident_mask] == y_pred_discrete[confident_mask]
            )
            metrics['confident_ratio'] = confident_mask.mean()
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize='true'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    return plt.gcf()


def analyze_variance_vs_error(y_true, y_pred, variance):
    """Analyze relationship between prediction variance and error"""
    errors = np.abs(y_true - y_pred)
    
    # Create error groups
    error_groups = []
    variances = []
    
    for i, error in enumerate(errors):
        if error < 0.2:
            error_groups.append('0')
        elif error < 0.4:
            error_groups.append('1')
        else:
            error_groups.append('2+')
        variances.append(variance[i])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = [0, 1, 2]
    labels = ['0', '1', '2+']
    
    for pos, label in zip(positions, labels):
        data = [v for v, g in zip(variances, error_groups) if g == label]
        bp = ax.boxplot(data, positions=[pos], labels=[label], widths=0.6)
    
    ax.set_xlabel('Absolute Error Group')
    ax.set_ylabel('Variance')
    ax.set_title('Prediction Variance vs Absolute Error')
    
    return fig