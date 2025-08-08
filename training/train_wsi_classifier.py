import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import tensorflow_addons as tfa

from models.wisdom import WiSDoMWSIClassifier
from training.losses import ordinal_regression_loss
from data.datasets.panda_dataset import load_wsi_dataset
from utils.prototypes import get_wsi_prototypes


def train_wsi_classifier(config):
    """Train WiSDoM WSI classifier"""
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_wsi_dataset(
        batch_size=config['batch_size']
    )
    
    # Get prototypes for initialization
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
        dim_y=config['dim_y'],
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
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
        loss=ordinal_regression_loss(alpha=config['alpha']),
        metrics=['mean_absolute_error',
                 tfa.metrics.CohenKappa(num_classes=6, weightage='quadratic')]
    )
    
    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        config['checkpoint_path'],
        monitor="val_mean_absolute_error",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error",
        patience=config['patience'],
        verbose=1,
        restore_best_weights=True,
        mode="min",
    )
    
    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['epochs'],
        callbacks=[checkpoint_callback, early_stopping],
        verbose=1
    )
    
    # Evaluate
    test_results = model.evaluate(test_dataset)
    
    return model, history, test_results


if __name__ == "__main__":
    config = {
        'batch_size': 4,
        'patch_size': 192,
        'image_size': 1152,
        'strides': 192,
        'encoded_size': 128,
        'dim_y': 6,
        'n_comp': 216,
        'n_prototypes': 216,
        'sigma': 1.0,
        'use_attention': True,
        'attention_dim_h': 64,
        'attention_dense_units_1': 128,
        'attention_dense_units_2': 128,
        'learning_rate': 1e-4,
        'alpha': 0.1,
        'epochs': 50,
        'patience': 5,
        'checkpoint_path': './checkpoints/wisdom_wsi_best.h5'
    }
    
    model, history, results = train_wsi_classifier(config)
    print(f"Test Results: {results}")