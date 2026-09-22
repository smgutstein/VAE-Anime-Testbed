import logging
import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout
from pathlib import Path
from utils import setup_logging



class Sampling(tf.keras.layers.Layer):
  def __init__(self, seed=0, **kwargs):
    super().__init__(**kwargs)
    self.base_seed = int(seed)
    # Training-position state for epoch/step-keyed sampling.  The current
    # sampling path does not read these yet, so setting them has no effect on
    # results.  Keep them out of model.weights so .weights.h5 files are
    # unaffected.
    self.epoch_seed = self._no_dependency(
        tf.Variable(0, dtype=tf.int32, trainable=False)
    )
    self.step_seed = self._no_dependency(
        tf.Variable(-1, dtype=tf.int32, trainable=False)
    )

  def set_training_position(self, epoch, step):
    self.epoch_seed.assign(int(epoch))
    self.step_seed.assign(int(step))

  def call(self, inputs):
    """Generates a random sample and combines with the encoder output
    
    Args:
      inputs -- output tensor from the encoder

    Returns:
      `inputs` tensors combined with a random sample
    """
    mu, log_var = inputs
    batch = tf.shape(mu)[0]
    dim = tf.shape(mu)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    z = mu + tf.exp(0.5 * log_var) * epsilon 

    return  z
  

    
class VAE_Encoder:
    def __init__(self, 
                 enc_input_shape=(64,64,3), 
                 latent_dim=512,
                 base_filters=32,
                 filter_factors=(1, 2, 4),
                 encode_dense_units=1024,
                 kernel_size=3,
                 output_dir="scratch_output",
                 random_seed=0):
        self.enc_input_shape = enc_input_shape
        self.base_filters = base_filters
        self.filter_factors = list(filter_factors)
        self.encode_dense_units = encode_dense_units
        self.k_size = kernel_size
        self.latent_dim = latent_dim
        self.output_dir = Path(output_dir)
        self.random_seed = int(random_seed)
        self.encoder_net = None
        self.num_input_pixels = np.prod(enc_input_shape)
                

    def encoder_layers(self, inputs):
        """Defines the encoder's layers.
        Args:
            inputs -- batch from the dataset
            latent_dim -- dimensionality of the latent space
        Returns:
            mu -- learned mean
            log_var -- learned log variance
            batch_3.shape -- shape of the features before flattening
        """
        x = tf.keras.layers.Conv2D(filters=self.base_filters*self.filter_factors[0], 
                                kernel_size=self.k_size, strides=2, 
                                padding="same", 
                                activation=None, name="encoder_conv1")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x= tf.keras.layers.LeakyReLU(name="lrelu_1")(x)


        x = tf.keras.layers.Conv2D(filters=self.base_filters*self.filter_factors[1], 
                                kernel_size=self.k_size, strides=2, 
                                padding='same', 
                                activation=None, name="encoder_conv2")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(name="lrelu_2")(x)
        
        x = tf.keras.layers.Conv2D(filters=self.base_filters*self.filter_factors[2], 
                                kernel_size=self.k_size, strides=2, 
                                padding='same', 
                                activation=None, name="encoder_conv3")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(name="lrelu_3")(x)

        # Flatten the features and feed into the Dense network
        # to get the mu and log_var. Important to note that the
        # Flatten layer is named 'encoder_flatten' for easy access
        # to the layer's input shape by the decoder.
        x = tf.keras.layers.Flatten(name="encoder_flatten")(x)

        x = tf.keras.layers.Dense(self.encode_dense_units, 
                                  activation=None, 
                                  name="encoder_dense")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(name="lrelu_4")(x)

        # add output Dense networks for mu and log_var, units equal 
        # to the declared latent_dim.
        mu = tf.keras.layers.Dense(self.latent_dim, name='latent_mu',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-3),
                                   bias_initializer=tf.keras.initializers.Zeros(),)(x)
        log_var = tf.keras.layers.Dense(self.latent_dim, name ='latent_log_var',
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-3),
                                        bias_initializer=tf.keras.initializers.Constant(-1.0),)(x)  

        return mu, log_var
    
    def set_encoder_model(self):
        """Defines the encoder model with the Sampling layer
        Args:
        latent_dim -- dimensionality of the latent space
        input_shape -- shape of the dataset batch

        Returns:
        model -- the encoder model
        conv_shape -- shape of the features before flattening
        """

        inputs = tf.keras.layers.Input(shape=self.enc_input_shape)
        mu, log_var = self.encoder_layers(inputs)
        self.sampling_layer = Sampling(seed=self.random_seed)
        z = self.sampling_layer((mu, log_var))
        self.encoder_net = tf.keras.Model(inputs, 
                                          outputs=[mu, log_var, z],
                                          name="Encoder_Model")
        
        return None
    
    def show_model(self):
        if self.encoder_net:
           with open(self.output_dir / Path('encoder_model_summary.txt'), 'w') as f:
                # Redirect stdout to the file
                with redirect_stdout(f):
                    self.encoder_net.summary()
                    logging.info("Model summary has been saved to 'model_summary.txt'")   
        else:
            logging.error("Encoder Model not yet defined")


if __name__ == '__main__':
    
    setup_logging()
    encoder = VAE_Encoder(enc_input_shape=(64,64,3,), 
                          latent_dim=512,
                          base_filters=32,
                          filter_factors=(1, 2, 4),
                          encode_dense_units=1024,
                          kernel_size=3)
    encoder.set_encoder_model()
    encoder.show_model()
    enc_output_shape = tuple(encoder.encoder_net.get_layer('encoder_flatten').input.shape)
    logging.info(f"Output shape of the encoder: {enc_output_shape}")