import tensorflow as tf
from tensorflow.keras import layers


class KQMAttentionLayer(tf.keras.layers.Layer):
    """Local-Global Attention for patch aggregation"""
    
    def __init__(self,
                 dim_h,
                 dense_units_1,
                 dense_units_2):
        super().__init__()
        self.dim_h = dim_h
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.mlp_1 = tf.keras.Sequential([
            layers.Dense(dense_units_1, activation='relu'),
            layers.Dense(dim_h, activation='linear')])
        self.mlp_2 = tf.keras.Sequential([
            layers.Dense(dense_units_2, activation='relu'),
            layers.Dense(1, activation='linear')])
    
    def call(self, input):
        z_local = self.mlp_1(input)
        z_global = tf.reduce_mean(z_local, axis=1)
        z_global = tf.expand_dims(z_global, axis=1)
        z_global = tf.broadcast_to(z_global, tf.shape(z_local))
        z = tf.concat([z_local, z_global], axis=-1)
        z = self.mlp_2(z)
        z = tf.squeeze(z, axis=-1)
        w = tf.nn.softmax(z)
        return w