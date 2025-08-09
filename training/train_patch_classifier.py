import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow_addons as tfa

from models.wisdom import WiSDoMPatchClassifier
from training.losses import ordinal_regression_loss, classification_loss
from data.datasets.panda_dataset import load_patch_dataset
from utils.prototypes import get_patch_prototypes


def train_patch_classifier(config, regression=False):
    """Train WiSDoM patch classifier
    
    Args:
        config: Configuration dictionary
        regression: If True, train for ordinal regression, else classification
    """
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_patch_dataset(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        regression=regression
    )
    
    # Get prototypes for initialization
    prototypes, prototype_labels = get_patch_prototypes(
        train_dataset,
        n_prototypes=config['n_prototypes']
    )
    
    # Create model
    model = WiSDoMPatchClassifier(
        encoded_size=config['encoded_size'],
        dim_y=config['dim_y'],  # 5 for patches (stroma, benign, G3, G4, G5)
        n_comp=config['n_comp'],
        sigma=config['sigma']
    )
    
    # Initialize components
    model.init_components(prototypes, prototype_labels, 
                         init_sigma=True, sigma_mult=1.0)
    print(f"Initialized sigma: {model.kernel.sigma.numpy():.4f}")
    
    # Compile model
    if regression:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
            loss=ordinal_regression_loss(alpha=config['alpha']),
            metrics=['mean_absolute_error',
                     tfa.metrics.CohenKappa(num_classes=5, weightage='quadratic')]
        )
        monitor_metric = "val_mean_absolute_error"
        mode = "min"
    else:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
            loss=classification_loss(),
            metrics=['categorical_accuracy',
                     tfa.metrics.CohenKappa(num_classes=5, weightage='quadratic')]
        )
        monitor_metric = "val_cohen_kappa"
        mode = "max"
    
    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        config['checkpoint_path'],
        monitor=monitor_metric,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode=mode,
        save_freq="epoch",
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=config['patience'],
        verbose=1,
        restore_best_weights=True,
        mode=mode,
    )
    
    # Warmup encoder (optional)
    if config.get('warmup_epochs', 0) > 0:
        print(f"Warming up encoder for {config['warmup_epochs']} epochs...")
        model.encoder.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
            loss='categorical_crossentropy' if not regression else 'mse',
            metrics=['accuracy'] if not regression else ['mae']
        )
        
        # Create warmup dataset with direct encoder training
        warmup_history = model.encoder.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config['warmup_epochs'],
            verbose=1
        )
    
    # Train full model
    print("Training full model...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        callbacks=[checkpoint_callback, early_stopping],
        verbose=1
    )
    
    # Evaluate
    print("Evaluating on test set...")
    test_results = model.evaluate(test_dataset)
    
    return model, history, test_results


def main():
    """Main training function for patch classification"""
    
    # Configuration for patch classification
    config_classification = {
        'data_dir': '/path/to/panda/dataset',
        'batch_size': 64,
        'encoded_size': 128,
        'dim_y': 5,  # 5 classes for patches
        'n_comp': 216,
        'n_prototypes': 216,
        'sigma': 0.1,
        'learning_rate': 1e-4,
        'alpha': 0.1,  # For regression only
        'epochs': 30,
        'warmup_epochs': 2,
        'patience': 5,
        'checkpoint_path': './checkpoints/wisdom_patch_classification.h5'
    }
    
    # Configuration for patch regression
    config_regression = config_classification.copy()
    config_regression['checkpoint_path'] = './checkpoints/wisdom_patch_regression.h5'
    
    # Train classification model
    print("Training patch classifier (classification)...")
    model_cls, history_cls, results_cls = train_patch_classifier(
        config_classification, regression=False
    )
    print(f"Classification Results: {results_cls}")
    
    # Train regression model
    print("\nTraining patch classifier (regression)...")
    model_reg, history_reg, results_reg = train_patch_classifier(
        config_regression, regression=True
    )
    print(f"Regression Results: {results_reg}")
    
    return {
        'classification': (model_cls, history_cls, results_cls),
        'regression': (model_reg, history_reg, results_reg)
    }
