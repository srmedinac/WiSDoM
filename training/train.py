import os
import json
import argparse
from datetime import datetime

from train_patch_classifier import train_patch_classifier
from train_wsi_regression import train_wsi_model


def train_all_models(base_config):
    """Train all WiSDoM model variants"""
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = os.path.join(base_config['results_dir'], timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Patch Classification
    print("\n" + "="*60)
    print("TRAINING PATCH CLASSIFICATION MODEL")
    print("="*60)
    patch_cls_config = base_config.copy()
    patch_cls_config['checkpoint_path'] = os.path.join(
        results_dir, 'patch_classification.h5'
    )
    patch_cls_config['dim_y'] = 5  # Patches have 5 classes
    
    model, history, test_results = train_patch_classifier(
        patch_cls_config, regression=False
    )
    results['patch_classification'] = {
        'test_results': test_results,
        'config': patch_cls_config
    }
    
    # 2. Patch Regression
    print("\n" + "="*60)
    print("TRAINING PATCH REGRESSION MODEL")
    print("="*60)
    patch_reg_config = base_config.copy()
    patch_reg_config['checkpoint_path'] = os.path.join(
        results_dir, 'patch_regression.h5'
    )
    patch_reg_config['dim_y'] = 5
    
    model, history, test_results = train_patch_classifier(
        patch_reg_config, regression=True
    )
    results['patch_regression'] = {
        'test_results': test_results,
        'config': patch_reg_config
    }
    
    # 3. WSI Classification
    print("\n" + "="*60)
    print("TRAINING WSI CLASSIFICATION MODEL")
    print("="*60)
    wsi_cls_config = base_config.copy()
    wsi_cls_config['checkpoint_path'] = os.path.join(
        results_dir, 'wsi_classification.h5'
    )
    wsi_cls_config['dim_y'] = 6  # WSI have 6 ISUP grades
    
    model, history, test_results = train_wsi_model(
        wsi_cls_config, regression=False
    )
    results['wsi_classification'] = {
        'test_results': test_results,
        'config': wsi_cls_config
    }
    
    # 4. WSI Regression
    print("\n" + "="*60)
    print("TRAINING WSI REGRESSION MODEL")
    print("="*60)
    wsi_reg_config = base_config.copy()
    wsi_reg_config['checkpoint_path'] = os.path.join(
        results_dir, 'wsi_regression.h5'
    )
    wsi_reg_config['dim_y'] = 6
    
    model, history, test_results = train_wsi_model(
        wsi_reg_config, regression=True
    )
    results['wsi_regression'] = {
        'test_results': test_results,
        'config': wsi_reg_config
    }
    
    # Save results summary
    with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train all WiSDoM models')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to PANDA dataset')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    
    args = parser.parse_args()
    
    # Base configuration for all models
    base_config = {
        'data_dir': args.data_dir,
        'results_dir': args.results_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        # Model architecture
        'patch_size': 192,
        'image_size': 1152,
        'strides': 192,
        'encoded_size': 128,
        'n_comp': 216,
        'n_prototypes': 216,
        'sigma': 1.0,
        # Attention (for WSI)
        'use_attention': True,
        'attention_dim_h': 64,
        'attention_dense_units_1': 128,
        'attention_dense_units_2': 128,
        # Training
        'learning_rate': 1e-4,
        'alpha': 0.1,  # Variance penalty
        'patience': 5,
        'warmup_epochs': 2,
        'variance_threshold': 0.05
    }
    
    results = train_all_models(base_config)
    
    # Print final summary
    print("\nFinal Results Summary:")
    print("-" * 40)
    for model_name, model_results in results.items():
        print(f"{model_name}:")
        print(f"  Test results: {model_results['test_results']}")
    

if __name__ == "__main__":
    main()