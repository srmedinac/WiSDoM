import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances

from .kdm.density_matrices import pure2dm, comp2dm, dm2discrete
from .kdm.kernels import RBFKernelLayer
from .kdm.kdm_units import KQMUnit
from .attention.local_global_attention import KQMAttentionLayer
from .encoders.convnext_encoder import create_convnext_encoder, PatchEncoder


class Patches(tf.keras.layers.Layer):
    """Extract patches from whole slide images"""
    
    def __init__(self, patch_size, image_size, strides):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.strides = strides
        self.num_patches = (image_size - patch_size) // strides + 1

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.strides, self.strides, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches ** 2, patch_dims])
        return patches


class ProbRegression(tf.keras.layers.Layer):
    """Calculate expected value and variance for ordinal regression"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('A `ProbRegression` layer should be '
                             'called with a tensor of shape '
                             '(batch_size, n)')
        self.vals = tf.constant(tf.linspace(0.0, 1.0, input_shape[1]), dtype=tf.float32)
        self.vals2 = self.vals ** 2
        self.built = True

    def call(self, inputs):
        mean = tf.einsum('...i,i->...', inputs, self.vals, optimize='optimal')
        mean2 = tf.einsum('...i,i->...', inputs, self.vals2, optimize='optimal')
        var = mean2 - mean ** 2
        return tf.stack([mean, var], axis=-1)


class WiSDoMPatchClassifier(tf.keras.Model):
    """Fully-supervised patch-level Gleason grading model"""
    
    def __init__(self,
                 encoded_size=128,
                 dim_y=5,
                 n_comp=216,
                 sigma=0.1):
        super().__init__()
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.n_comp = n_comp
        
        # Create encoder
        convnext = create_convnext_encoder(encoded_size)
        self.encoder = convnext
        
        # Create KDM components
        self.kernel = RBFKernelLayer(sigma=sigma,
                                     dim=encoded_size,
                                     trainable=True)
        self.kqm_unit = KQMUnit(kernel=self.kernel,
                               dim_x=encoded_size,
                               dim_y=dim_y,
                               n_comp=n_comp)
        self.regression_layer = ProbRegression()

    def call(self, input):
        encoded = self.encoder(input)
        rho_x = pure2dm(encoded)
        rho_y = self.kqm_unit(rho_x)
        probs = dm2discrete(rho_y)
        mean_var = self.regression_layer(probs)
        return mean_var

    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1):
        """Initialize KDM components with prototypes"""
        encoded_x = self.encoder(samples_x)
        if init_sigma:
            distances = pairwise_distances(encoded_x)
            sigma = np.mean(distances) * sigma_mult
            self.kernel.sigma.assign(sigma)
        self.kqm_unit.c_x.assign(encoded_x)
        self.kqm_unit.c_y.assign(samples_y)
        self.kqm_unit.comp_w.assign(tf.ones((self.n_comp,)) / self.n_comp)


class WiSDoMWSIClassifier(tf.keras.Model):
    """Weakly-supervised whole-slide ISUP grading model"""
    
    def __init__(self,
                 patch_size=192,
                 image_size=1152,
                 strides=192,
                 encoded_size=128,
                 dim_y=6,
                 n_comp=216,
                 sigma=1.0,
                 attention=True,
                 attention_dim_h=64,
                 attention_dense_units_1=128,
                 attention_dense_units_2=128):
        super().__init__()
        
        self.patch_size = patch_size
        self.image_size = image_size
        self.strides = strides
        self.dim_y = dim_y
        self.encoded_size = encoded_size
        self.n_comp = n_comp
        self.attention = attention
        
        # Create layers
        self.patch_layer = Patches(patch_size, image_size, strides)
        
        # Create encoder
        convnext = create_convnext_encoder(encoded_size)
        self.encoder = PatchEncoder(convnext, encoded_size)
        
        # Create KDM components
        self.kernel = RBFKernelLayer(sigma=sigma,
                                     dim=encoded_size,
                                     trainable=True)
        self.kqm_unit = KQMUnit(kernel=self.kernel,
                               dim_x=encoded_size,
                               dim_y=dim_y,
                               n_comp=n_comp)
        
        # Create attention layer if needed
        if attention:
            self.attention_layer = KQMAttentionLayer(
                dim_h=attention_dim_h,
                dense_units_1=attention_dense_units_1,
                dense_units_2=attention_dense_units_2)
        
        self.regression_layer = ProbRegression()

    def call(self, input):
        # Extract patches
        patches = self.patch_layer(input)
        
        # Encode patches
        encoded = self.encoder(patches)
        bs = tf.shape(encoded)[0]
        
        # Calculate attention weights
        if self.attention:
            w = self.attention_layer(encoded)
        else:
            w = tf.ones((bs, self.patch_layer.num_patches ** 2,)) / (self.patch_layer.num_patches ** 2)
        
        # Create density matrix and perform inference
        rho_x = comp2dm(w, encoded)
        rho_y = self.kqm_unit(rho_x)
        probs = dm2discrete(rho_y)
        mean_var = self.regression_layer(probs)
        
        return mean_var

    def init_components(self, samples_x, samples_y, init_sigma=False, sigma_mult=1):
        """Initialize KDM components with prototypes"""
        patches = self.patch_layer(samples_x)
        idx = tf.random.uniform(shape=(patches.shape[0],), maxval=patches.shape[1], dtype=tf.int32)
        selected_patches = tf.gather(patches, idx, axis=1, batch_dims=1)
        encoded_x = self.encoder(selected_patches[:, tf.newaxis, :])[:, 0, :]
        
        if init_sigma:
            distances = pairwise_distances(encoded_x)
            sigma = np.mean(distances) * sigma_mult
            self.kernel.sigma.assign(sigma)
        
        self.kqm_unit.c_x.assign(encoded_x)
        self.kqm_unit.c_y.assign(samples_y)
        self.kqm_unit.comp_w.assign(tf.ones((self.n_comp,)) / self.n_comp)

    def visualize_attention(self, input):
        """Generate attention heatmap for input WSI"""
        patches = self.patch_layer(input)
        encoded = self.encoder(patches)
        w = self.attention_layer(encoded)
        
        conv2dt = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=self.patch_layer.patch_size,
            strides=self.patch_layer.strides,
            kernel_initializer=tf.keras.initializers.Ones(),
            bias_initializer=tf.keras.initializers.Zeros(),
            trainable=False)
        
        w = tf.reshape(w, [-1,
                          self.patch_layer.num_patches,
                          self.patch_layer.num_patches, 1])
        out = conv2dt(w)
        return out