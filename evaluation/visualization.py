import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize


def visualize_attention_heatmap(model, image, save_path=None):
    """Generate and visualize attention heatmap"""
    
    heatmap = model.visualize_attention(image)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    axes[0].imshow(image[0] if len(image.shape) == 4 else image)
    axes[0].set_title('Original WSI')
    axes[0].axis('off')
    
    # Attention heatmap
    im = axes[1].imshow(heatmap[0, :, :, 0], cmap='hot', alpha=0.7)
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_prototype_space(prototypes, labels=None, title="Prototype Space"):
    """Visualize learned prototypes using t-SNE"""
    
    # Extract active prototypes
    active_prototypes = prototypes['active_c_x']
    active_labels = np.argmax(prototypes['active_c_y'], axis=1) if labels is None else labels
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(active_prototypes)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], 
                         c=active_labels, cmap='viridis',
                         s=100, alpha=0.7, edgecolors='w')
    plt.colorbar(scatter, label='Class')
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Add prototype weights as size
    sizes = prototypes['active_weights'] * 5000
    plt.scatter(embedded[:, 0], embedded[:, 1], 
               s=sizes, alpha=0.3, c='red', marker='o')
    
    return plt.gcf()


def plot_gleason_pattern_distribution(predictions, true_labels, grade_group):
    """Plot Gleason pattern distribution for a specific ISUP grade group"""
    
    patterns = {
        'Stroma': [],
        'Benign': [],
        'Gleason 3': [],
        'Gleason 4': [],
        'Gleason 5': []
    }
    
    # Filter by grade group
    mask = true_labels == grade_group
    group_predictions = predictions[mask]
    
    # Calculate average distribution
    avg_distribution = np.mean(group_predictions, axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pattern_names = list(patterns.keys())
    x_pos = np.arange(len(pattern_names))
    
    bars = ax.bar(x_pos, avg_distribution)
    ax.set_xlabel('Tissue Pattern')
    ax.set_ylabel('Average Percentage')
    ax.set_title(f'Gleason Pattern Distribution - ISUP Grade Group {grade_group}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pattern_names)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_distribution):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}', ha='center', va='bottom')
    
    return fig


def plot_uncertainty_map(image, predictions, variances, threshold=0.05):
    """Create uncertainty visualization map"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image if len(image.shape) == 3 else image[0])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predictions
    axes[1].imshow(image if len(image.shape) == 3 else image[0])
    axes[1].set_title('Predictions')
    axes[1].axis('off')
    
    # Uncertainty map
    uncertainty_map = variances.reshape(image.shape[:2]) if len(variances.shape) == 1 else variances
    
    # Create colored overlay
    high_uncertainty = uncertainty_map > threshold
    overlay = np.zeros((*uncertainty_map.shape, 4))
    overlay[high_uncertainty] = [1, 0, 0, 0.5]  # Red with transparency
    overlay[~high_uncertainty] = [0, 0, 1, 0.3]  # Blue with transparency
    
    axes[2].imshow(image if len(image.shape) == 3 else image[0])
    axes[2].imshow(overlay)
    axes[2].set_title('Uncertainty Map (Red=High, Blue=Low)')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def create_prediction_summary(model, test_dataset, num_samples=5):
    """Create comprehensive prediction summary visualization"""
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, num_samples * 5))
    
    sample_idx = 0
    for batch_x, batch_y in test_dataset:
        if sample_idx >= num_samples:
            break
            
        for x, y in zip(batch_x, batch_y):
            if sample_idx >= num_samples:
                break
            
            # Get prediction
            pred = model(tf.expand_dims(x, 0))
            mean = pred[0, 0].numpy()
            var = pred[0, 1].numpy()
            
            # Original image
            axes[sample_idx, 0].imshow(x)
            axes[sample_idx, 0].set_title(f'True: {y.numpy():.2f}')
            axes[sample_idx, 0].axis('off')
            
            # Attention heatmap
            if hasattr(model, 'visualize_attention'):
                attn = model.visualize_attention(tf.expand_dims(x, 0))
                axes[sample_idx, 1].imshow(attn[0, :, :, 0], cmap='hot')
                axes[sample_idx, 1].set_title('Attention')
            else:
                axes[sample_idx, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[sample_idx, 1].set_title('Attention')
            axes[sample_idx, 1].axis('off')
            
            # Prediction
            axes[sample_idx, 2].text(0.5, 0.5, 
                                     f'Pred: {mean:.2f}\nVar: {var:.4f}',
                                     ha='center', va='center', fontsize=14)
            axes[sample_idx, 2].set_title('Prediction')
            axes[sample_idx, 2].axis('off')
            
            # Error
            error = abs(mean - y.numpy())
            color = 'green' if error < 0.2 else 'orange' if error < 0.4 else 'red'
            axes[sample_idx, 3].text(0.5, 0.5, 
                                     f'Error: {error:.3f}',
                                     ha='center', va='center', fontsize=14,
                                     color=color)
            axes[sample_idx, 3].set_title('Error')
            axes[sample_idx, 3].axis('off')
            
            sample_idx += 1
    
    plt.tight_layout()
    return fig