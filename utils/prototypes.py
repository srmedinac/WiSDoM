import numpy as np
import tensorflow as tf
from tqdm import tqdm


def get_samples_from_each_class(dataset, n_samples_per_class):
    """
    Extract stratified samples from dataset for prototype initialization
    
    Args:
        dataset: TensorFlow dataset
        n_samples_per_class: Number of samples per class
    
    Returns:
        Tuple of (samples, labels)
    """
    samples_per_class = {class_idx: [] for class_idx in range(6)}
    samples_found_per_class = {class_idx: 0 for class_idx in range(6)}
    samples_collected = 0

    for batch_samples, batch_labels in dataset:
        for sample, label in zip(batch_samples, batch_labels):
            if len(label.shape) > 0:  # One-hot encoded
                class_idx = np.argmax(label)
            else:  # Continuous label
                class_idx = int(np.round(label * 5))
            
            if samples_found_per_class[class_idx] < n_samples_per_class:
                samples_per_class[class_idx].append(sample)
                samples_found_per_class[class_idx] += 1
                samples_collected += 1

            if samples_collected == n_samples_per_class * 6:
                break

        if samples_collected == n_samples_per_class * 6:
            break

    # Stack samples
    stacked_samples = tf.stack([sample for samples_list in samples_per_class.values() 
                                for sample in samples_list])
    
    # Create labels
    if len(batch_labels.shape) > 1:  # One-hot encoded
        stacked_labels = tf.stack([tf.one_hot(class_idx, depth=6) 
                                   for class_idx, samples_list in samples_per_class.items() 
                                   for _ in range(len(samples_list))])
    else:  # Continuous
        stacked_labels = tf.stack([class_idx / 5.0 
                                   for class_idx, samples_list in samples_per_class.items() 
                                   for _ in range(len(samples_list))])

    return stacked_samples, stacked_labels


def get_wsi_prototypes(dataset, n_prototypes=216):
    """Get prototypes for WSI classification (36 per class)"""
    n_per_class = n_prototypes // 6
    return get_samples_from_each_class(dataset, n_per_class)


def get_patch_prototypes(dataset, n_prototypes=216):
    """Get prototypes for patch classification"""
    # 5 classes for patches (stroma, benign, G3, G4, G5)
    n_per_class = n_prototypes // 5
    prototypes = []
    labels = []
    
    for class_idx in range(5):
        class_samples = []
        for batch_x, batch_y in dataset:
            for x, y in zip(batch_x, batch_y):
                if len(y.shape) > 0:
                    current_class = np.argmax(y)
                else:
                    current_class = int(np.round(y * 4))
                
                if current_class == class_idx:
                    class_samples.append(x)
                    if len(class_samples) >= n_per_class:
                        break
            
            if len(class_samples) >= n_per_class:
                break
        
        prototypes.extend(class_samples[:n_per_class])
        if len(batch_y.shape) > 1:
            labels.extend([tf.one_hot(class_idx, depth=5)] * n_per_class)
        else:
            labels.extend([class_idx / 4.0] * n_per_class)
    
    # Add remaining prototypes if needed
    remaining = n_prototypes - len(prototypes)
    if remaining > 0:
        for batch_x, batch_y in dataset:
            for x, y in zip(batch_x, batch_y):
                prototypes.append(x)
                labels.append(y)
                remaining -= 1
                if remaining == 0:
                    break
            if remaining == 0:
                break
    
    return tf.stack(prototypes), tf.stack(labels)


def extract_learned_prototypes(model):
    """
    Extract learned prototypes from trained model
    
    Args:
        model: Trained WiSDoM model
    
    Returns:
        Dictionary containing prototype information
    """
    prototypes = {
        'c_x': model.kqm_unit.c_x.numpy(),
        'c_y': model.kqm_unit.c_y.numpy(),
        'weights': model.kqm_unit.comp_w.numpy()
    }
    
    # Normalize weights
    prototypes['weights'] = prototypes['weights'] / prototypes['weights'].sum()
    
    # Find active prototypes (weight > threshold)
    active_mask = prototypes['weights'] > 0.01
    prototypes['active_indices'] = np.where(active_mask)[0]
    prototypes['active_c_x'] = prototypes['c_x'][active_mask]
    prototypes['active_c_y'] = prototypes['c_y'][active_mask]
    prototypes['active_weights'] = prototypes['weights'][active_mask]
    
    return prototypes


def find_nearest_samples(prototypes, dataset, encoder, top_k=3):
    """
    Find nearest training samples to learned prototypes
    
    Args:
        prototypes: Dictionary from extract_learned_prototypes
        dataset: Training dataset
        encoder: Model encoder
        top_k: Number of nearest samples to return
    
    Returns:
        Dictionary mapping prototype indices to nearest samples
    """
    from sklearn.metrics import pairwise_distances
    
    # Collect encoded features from dataset
    all_features = []
    all_samples = []
    
    for batch_x, _ in dataset:
        encoded = encoder(batch_x)
        all_features.append(encoded.numpy())
        all_samples.append(batch_x.numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    all_samples = np.concatenate(all_samples, axis=0)
    
    # Compute distances
    distances = pairwise_distances(prototypes['active_c_x'], all_features)
    
    # Find nearest samples
    nearest_samples = {}
    for i, proto_idx in enumerate(prototypes['active_indices']):
        nearest_indices = np.argsort(distances[i])[:top_k]
        nearest_samples[proto_idx] = {
            'samples': all_samples[nearest_indices],
            'distances': distances[i][nearest_indices],
            'label': np.argmax(prototypes['active_c_y'][i])
        }
    
    return nearest_samples