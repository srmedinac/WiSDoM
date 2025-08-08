import tensorflow as tf


def ordinal_regression_loss(alpha=0.1):
    """
    Loss function for ordinal regression with variance penalty
    
    Args:
        alpha: Variance penalty weight
    """
    def loss(y_true, y_pred):
        mean = y_pred[:, 0:1]
        var = y_pred[:, 1:2]
        mse = tf.keras.losses.mean_squared_error(y_true, mean)
        return mse + alpha * var
    
    return loss


def classification_loss():
    """Standard categorical crossentropy loss"""
    return tf.keras.losses.CategoricalCrossentropy()