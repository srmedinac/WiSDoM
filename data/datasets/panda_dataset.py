import os
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder


def load_panda_metadata(data_dir):
    """Load PANDA dataset metadata"""
    
    # Load CSV files
    df_train = pd.read_csv(os.path.join(data_dir, "wsi_train.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "wsi_val.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "wsi_test.csv"))
    
    # Convert to dictionaries
    train_dict = df_train.set_index('image_id')['isup_grade'].to_dict(into=OrderedDict)
    val_dict = df_val.set_index('image_id')['isup_grade'].to_dict(into=OrderedDict)
    test_dict = df_test.set_index('image_id')['isup_grade'].to_dict(into=OrderedDict)
    
    # Filter for existing files
    mosaic_dir = os.path.join(data_dir, 'tile_mosaics')
    train_dict = {k: v for k, v in train_dict.items() 
                  if os.path.isfile(os.path.join(mosaic_dir, k+'.jpeg'))}
    val_dict = {k: v for k, v in val_dict.items() 
                if os.path.isfile(os.path.join(mosaic_dir, k+'.jpeg'))}
    test_dict = {k: v for k, v in test_dict.items() 
                 if os.path.isfile(os.path.join(mosaic_dir, k+'.jpeg'))}
    
    return train_dict, val_dict, test_dict


def create_wsi_dataset(paths, labels, batch_size=4, shuffle=False, 
                       regression=False):
    """Create TensorFlow dataset for WSI classification"""
    
    def decode_img(img):
        img = tf.io.decode_jpeg(img, channels=3)
        return img

    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label
    
    num_samples = len(labels)
    
    if regression:
        # Normalize labels to [0, 1] for ordinal regression
        labels = labels / 5.0
    else:
        # One-hot encode for classification
        encoder = OneHotEncoder()
        labels = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
    
    paths = tf.data.Dataset.from_tensor_slices(paths)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((paths, labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=num_samples)
    
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


def load_wsi_dataset(data_dir, batch_size=4, regression=True):
    """Load complete WSI dataset"""
    
    # Load metadata
    train_dict, val_dict, test_dict = load_panda_metadata(data_dir)
    
    # Extract features and labels
    train_features = list(train_dict.keys())
    train_labels = np.array(list(train_dict.values()))
    val_features = list(val_dict.keys())
    val_labels = np.array(list(val_dict.values()))
    test_features = list(test_dict.keys())
    test_labels = np.array(list(test_dict.values()))
    
    # Create paths
    mosaic_dir = os.path.join(data_dir, 'tile_mosaics')
    train_paths = [os.path.join(mosaic_dir, f+'.jpeg') for f in train_features]
    val_paths = [os.path.join(mosaic_dir, f+'.jpeg') for f in val_features]
    test_paths = [os.path.join(mosaic_dir, f+'.jpeg') for f in test_features]
    
    # Create datasets
    train_dataset = create_wsi_dataset(train_paths, train_labels, 
                                       batch_size, shuffle=True, 
                                       regression=regression)
    val_dataset = create_wsi_dataset(val_paths, val_labels, 
                                     batch_size, shuffle=False, 
                                     regression=regression)
    test_dataset = create_wsi_dataset(test_paths, test_labels, 
                                      batch_size, shuffle=False, 
                                      regression=regression)
    
    return train_dataset, val_dataset, test_dataset


def load_patch_dataset(data_dir, batch_size=64, regression=True):
    """Load patch-level dataset"""
    
    df_train = pd.read_csv(os.path.join(data_dir, "df_train.csv"))
    df_val = pd.read_csv(os.path.join(data_dir, "df_val.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "df_test.csv"))
    
    # Adjust Gleason scores
    df_train['gleason_score'] -= 1
    df_val['gleason_score'] -= 1
    df_test['gleason_score'] -= 1
    
    patches_folder = os.path.join(data_dir, "patches")
    
    def create_patch_dataset(df, shuffle=False):
        patches_path = df['image_id'].apply(
            lambda x: os.path.join(patches_folder, x.split('_')[0], x + '.jpeg')
        )
        
        if regression:
            labels = df['gleason_score'] / 4
        else:
            labels = tf.keras.utils.to_categorical(df['gleason_score'], num_classes=5)
        
        return create_wsi_dataset(patches_path.values, labels, 
                                 batch_size, shuffle, regression)
    
    train_dataset = create_patch_dataset(df_train, shuffle=True)
    val_dataset = create_patch_dataset(df_val, shuffle=False)
    test_dataset = create_patch_dataset(df_test, shuffle=False)
    
    return train_dataset, val_dataset, test_dataset