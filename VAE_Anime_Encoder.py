import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout
from pathlib import Path


class Sampling(tf.keras.layers.Layer):
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
                 output_dir="scratch_output"):
        self.enc_input_shape = enc_input_shape
        self.base_filters = 32
        self.filter_factors = [1,2,4]
        self.encode_dense_units = 1024
        self.k_size = 3
        self.latent_dim = latent_dim
        self.output_dir = Path(output_dir)
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
                                activation='relu', name="encoder_conv1")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x= tf.keras.layers.LeakyReLU(name="lrelu_1")(x)


        x = tf.keras.layers.Conv2D(filters=self.base_filters*self.filter_factors[1], 
                                kernel_size=self.k_size, strides=2, 
                                padding='same', 
                                activation='relu', name="encoder_conv2")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(name="lrelu_2")(x)
        
        x = tf.keras.layers.Conv2D(filters=self.base_filters*self.filter_factors[2], 
                                kernel_size=self.k_size, strides=2, 
                                padding='same', 
                                activation='relu', name="encoder_conv3")(x)

        # assign to a different variable so you can extract the shape later
        x = tf.keras.layers.BatchNormalization(name="last_batch_normalization")(x)
        x = tf.keras.layers.LeakyReLU(name="lrelu_3")(x)

        # flatten the features and feed into the Dense network
        x = tf.keras.layers.Flatten(name="encoder_flatten")(x)

        # we arbitrarily used 20 units here but feel free to change and 
        # see what results you get
        x = tf.keras.layers.Dense(self.encode_dense_units, 
                                  activation='relu', 
                                  name="encoder_dense")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(name="lrelu_4")(x)

        # add output Dense networks for mu and log_var, units equal 
        # to the declared latent_dim.
        mu = tf.keras.layers.Dense(self.latent_dim, name='latent_mu')(x)
        log_var = tf.keras.layers.Dense(self.latent_dim, name ='latent_log_var')(x)  

        # revise `batch_3.shape` here if you opted not to use 3 Conv2D layers
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
        z = Sampling()((mu, log_var))
        self.encoder_net = tf.keras.Model(inputs, 
                                          outputs=[mu, log_var, z],
                                          name="Encoder_Model")
        
        return None
    
    def show_model(self):
        if self.encoder_net:
           with open(self.output_dir / Path('encoder_model_summary.txt'), 'w') as f:
                # Redirect stdout to the file
                with redirect_stdout(f):
                    # Call model.summary(), which will now print to the file
                    self.encoder_net.summary()
                    print("Model summary has been saved to 'model_summary.txt'")   
        else:
            print("Encoder Model not yet defined")


if __name__ == '__main__':
    
    encoder = VAE_Encoder(enc_input_shape=(64,64,3,), 
                          latent_dim=512)
    encoder.set_encoder_model()
    encoder.show_model()
    enc_output_shape = encoder.encoder_net.get_layer('last_batch_normalization').input_shape
    print(f"Output shape of the encoder: {enc_output_shape}")