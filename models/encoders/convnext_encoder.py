import tensorflow as tf
from tensorflow.keras import layers, models


def create_convnext_encoder(encoded_size=128, input_shape=(192, 192, 3)):
    """Create ConvNeXT encoder for patch feature extraction"""
    
    convnext = tf.keras.applications.ConvNeXTTiny(
        model_name='convnext_tiny',
        include_top=False,
        include_preprocessing=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling="avg",
        classes=6,
        classifier_activation='softmax'
    )
    
    encoder = models.Sequential([
        layers.Input(shape=input_shape),
        convnext,
        layers.Dropout(0.5),
        layers.Dense(encoded_size, activation="relu"),
    ])
    
    return encoder


class PatchEncoder(tf.keras.Model):
    """Encoder wrapper for batch processing"""
    
    def __init__(self, encoder, encoded_size=128):
        super().__init__()
        self.encoded_size = encoded_size
        self.encoder = encoder

    def call(self, input):
        """
        Args:
            input: (bs, n_patches, w*h*c)
        Returns:
            (bs, num_patches, encoded_size)
        """
        bs = tf.shape(input)[0]
        input = tf.reshape(input, [-1, 192, 192, 3])
        x = self.encoder(input)
        out = tf.reshape(x, [bs, -1, self.encoded_size])
        return out