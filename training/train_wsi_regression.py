import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow_addons as tfa

from models.wisdom import WiSDoMWSIClassifier
from training.losses import ordinal_regression_loss, classification_loss
from data.datasets.panda_dataset import load_wsi_dataset
from utils.prototypes import get_wsi_prototypes
from evaluation.metrics import evaluate_predictions


def train_wsi_model(config, regression=True):
    """Train WiSDoM WSI model for classification or regression
    
    Args:
        config: Configuration dictionary
        regression: If True, train for ordinal regression, else classification
    """
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_wsi_dataset(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        regression=regression
    )
    
    # Get prototypes for initialization
    print("Extracting prototypes...")
    prototypes, prototype_labels = get_wsi_prototypes(
        train_dataset,
        n_prototypes=config['n_prototypes']
    )
    
    # Create model
    model = WiSDoMWSIClassifier(
        patch_size=config['patch_size'],
        image_size=config['image_size'],
        strides=config['strides'],
        encoded_size=config['encoded_size'],
        dim_y=config['dim_y'],  # 6 for ISUP grades
        n_comp=config['n_comp'],
        sigma=config['sigma'],
        attention=config['use_attention'],
        attention_dim_h=config['attention_dim_h'],
        attention_dense_units_1=config['attention_dense_units_1'],
        attention_dense_units_2=config['attention_dense_units_2']
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
                     tfa.metrics.CohenKappa(num_classes=6, weightage='quadratic')]
        )
        monitor_metric = "val_mean_absolute_error"
        mode = "min"
    else:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
            loss=classification_loss(),
            metrics=['categorical_accuracy',
                     tfa.metrics.CohenKappa(num_classes=6, weightage='quadratic')]
        )
        monitor_metric = "val_cohen_kappa"
        mode = "max"
    
    # Learning rate schedule
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
        mode=mode
    )
    
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
    
    # Warmup encoder if specified
    if config.get('warmup_epochs', 0) > 0 and config.get('warmup_weights', None):
        print(f"Loading pretrained encoder weights from {config['warmup_weights']}...")
        model.encoder.load_weights(config['warmup_weights'])
    
    # Train
    print(f"Training WSI model ({'regression' if regression else 'classification'})...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        callbacks=[checkpoint_callback, early_stopping, lr_schedule],
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(test_dataset)
    
    # Additional evaluation with variance filtering for regression
    if regression:
        print("\nEvaluating with variance filtering...")
        all_preds = []
        all_true = []
        
        for x_batch, y_batch in test_dataset:
            preds = model.predict(x_batch, verbose=0)
            all_preds.append(preds)
            all_true.append(y_batch.numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        
        # Extract mean and variance
        y_pred = all_preds[:, 0]
        variance = all_preds[:, 1]
        
        # Evaluate with variance filtering
        metrics = evaluate_predictions(
            all_true, y_pred, variance, 
            threshold=config.get('variance_threshold', 0.05)
        )
        
        print(f"Full test set - Kappa: {metrics['kappa']:.3f}, MAE: {metrics['mae']:.3f}")
        if 'confident_kappa' in metrics:
            print(f"Confident predictions ({metrics['confident_ratio']:.1%}) - "
                  f"Kappa: {metrics['confident_kappa']:.3f}, "
                  f"MAE: {metrics['confident_mae']:.3f}")
    
    return model, history, test_results


def main():
    """Main training function for WSI models"""
    
    # Base configuration
    base_config = {
        'data_dir': '/path/to/panda/dataset',
        'batch_size': 4,
        'patch_size': 192,
        'image_size': 1152,
        'strides': 192,
        'encoded_size': 128,
        'dim_y': 6,  # 6 ISUP grades
        'n_comp': 216,
        'n_prototypes': 216,
        'sigma': 1.0,
        'use_attention': True,
        'attention_dim_h': 64,
        'attention_dense_units_1': 128,
        'attention_dense_units_2': 128,
        'learning_rate': 1e-4,
        'alpha': 0.1,  # Variance penalty for regression
        'epochs': 50,
        'patience': 5,
        'warmup_epochs': 2,
        'warmup_weights': None,  # Path to pretrained encoder weights
        'variance_threshold': 0.05
    }
    
    # Configuration for regression
    config_regression = base_config.copy()
    config_regression['checkpoint_path'] = './checkpoints/wisdom_wsi_regression.h5'
    
    # Configuration for classification
    config_classification = base_config.copy()
    config_classification['checkpoint_path'] = './checkpoints/wisdom_wsi_classification.h5'
    
    # Train regression model
    print("="*50)
    print("Training WSI Regression Model")
    print("="*50)
    model_reg, history_reg, results_reg = train_wsi_model(
        config_regression, regression=True
    )
    
    # Train classification model
    print("\n" + "="*50)
    print("Training WSI Classification Model")
    print("="*50)
    model_cls, history_cls, results_cls = train_wsi_model(
        config_classification, regression=False
    )
    
    print("\n" + "="*50)
    print("Final Results Summary")
    print("="*50)
    print(f"Regression - Test Results: {results_reg}")
    print(f"Classification - Test Results: {results_cls}")
    
    return {
        'regression': (model_reg, history_reg, results_reg),
        'classification': (model_cls, history_cls, results_cls)
    }
