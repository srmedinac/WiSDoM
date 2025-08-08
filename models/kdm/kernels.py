import numpy as np
import tensorflow as tf


class RBFKernelLayer(tf.keras.layers.Layer):
    """RBF Kernel for KDM operations"""
    
    def __init__(self, sigma, dim, trainable=True, min_sigma=1e-3):
        super(RBFKernelLayer, self).__init__()
        if type(sigma) is tf.Variable:
            self.sigma = sigma
        else:
            self.sigma = tf.Variable(sigma, dtype=tf.float32, trainable=trainable)
        self.dim = dim
        self.min_sigma = min_sigma

    def call(self, A, B):
        """
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        """
        shape_A = tf.shape(A)
        shape_B = tf.shape(B)
        A_norm = tf.norm(A, axis=-1)[..., tf.newaxis] ** 2
        B_norm = tf.norm(B, axis=-1)[tf.newaxis, tf.newaxis, :] ** 2
        A_reshaped = tf.reshape(A, [-1, shape_A[2]])
        AB = tf.matmul(A_reshaped, B, transpose_b=True)
        AB = tf.reshape(AB, [shape_A[0], shape_A[1], shape_B[0]])
        dist2 = A_norm + B_norm - 2. * AB
        dist2 = tf.clip_by_value(dist2, 0., np.inf)
        sigma = tf.clip_by_value(self.sigma, self.min_sigma, np.inf)
        K = tf.exp(-dist2 / (2. * sigma ** 2.))
        return K

    def log_weight(self):
        sigma = tf.clip_by_value(self.sigma, self.min_sigma, np.inf)
        return - self.dim * tf.math.log(sigma + 1e-12) - self.dim * np.log(4 * np.pi)


class CosineKernelLayer(tf.keras.layers.Layer):
    """Cosine Kernel for KDM operations"""
    
    def __init__(self):
        super(CosineKernelLayer, self).__init__()
        self.eps = 1e-6

    def call(self, A, B):
        """
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
        """
        A = tf.math.divide_no_nan(A,
                                  tf.expand_dims(tf.norm(A, axis=-1), axis=-1))
        B = tf.math.divide_no_nan(B,
                                  tf.expand_dims(tf.norm(B, axis=-1), axis=-1))
        K = tf.einsum("...nd,md->...nm", A, B)
        return K

    def log_weight(self):
        return 0