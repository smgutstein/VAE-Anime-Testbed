import tensorflow as tf

from contextlib import redirect_stdout
from pathlib import Path    

class VAE_Decoder:
    def __init__(self, 
                 dec_input_shape, 
                 latent_dim, 
                 output_dir="scratch_output"):
        self.dec_input_shape = dec_input_shape
        self.base_filters = 32
        self.filter_factors = [1,2,4]
        self.k_size = 3
        self.latent_dim = latent_dim
        self.output_dir = Path(output_dir)
        self.decoder_net = None


    def decoder_layers(self, inputs):
        """Defines the decoder layers.
        Args:
            inputs -- output of the encoder 
            conv_shape -- shape of the features before flattening

        Returns:
            tensor containing the decoded output
        """

        units = self.dec_input_shape[1] * self.dec_input_shape[2] * self.dec_input_shape[3]
        x = tf.keras.layers.Dense(units, activation = 'relu', 
                                name="decoder_dense1")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # reshape output using the conv_shape dimensions
        x = tf.keras.layers.Reshape((self.dec_input_shape[1], 
                                    self.dec_input_shape[2],
                                    self.dec_input_shape[3]), 
                                    name="decoder_reshape")(x)

        # upsample the features back to the original dimensions
        x = tf.keras.layers.Conv2DTranspose(filters=self.base_filters*self.filter_factors[2], 
                                            kernel_size=self.k_size, 
                                            strides=2, padding='same', 
                                            activation='relu',  
                                            name="decoder_conv2d_2")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(name="lrelu_d1")(x)

        x = tf.keras.layers.Conv2DTranspose(filters=self.base_filters*self.filter_factors[1], 
                                            kernel_size=self.k_size, 
                                            strides=2, padding='same',
                                            activation='relu', 
                                            name="decoder_conv2d_3")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(name="lrelu_d2")(x)

        x = tf.keras.layers.Conv2DTranspose(filters=self.base_filters*self.filter_factors[0], 
                                            kernel_size=self.k_size, 
                                            strides=2, padding='same', 
                                            activation='relu', 
                                            name="decoder_conv2d_4")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(name="lrelu_d3")(x)

        
        x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, 
                                            padding='same', activation='sigmoid',
                                            name="decoder_final")(x)  
        return x    
    
    def set_decoder_model(self, decoder_input_shape=(None, 512)):
        """Defines the decoder model.
        Args:
            latent_dim -- dimensionality of the latent space
            conv_shape -- shape of the features before flattening

        Returns:
            model -- the decoder model
        """
        inputs = tf.keras.layers.Input(shape=(self.latent_dim))
        outputs = self.decoder_layers(inputs)
        self.decoder_net = tf.keras.Model(inputs, outputs,
                                          name="Decoder_Model")

        return None

    def show_model(self):
        if self.decoder_net:
           with open(self.output_dir / Path('decoder_model_summary.txt'), 'w') as f:
                # Redirect stdout to the file
                with redirect_stdout(f):
                    # Call model.summary(), which will now print to the file
                    self.decoder_net.summary()
                    print("Model summary has been saved to 'model_summary.txt'")   
        else:
            print("Decoder Model not yet defined")
if __name__ == '__main__':
    
    decoder = VAE_Decoder(dec_input_shape=(None, 8, 8, 128), 
                          latent_dim=512)
    decoder.set_decoder_model()
    decoder.show_model()