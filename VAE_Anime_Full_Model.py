from contextlib import redirect_stdout
import logging
import tensorflow as tf

from pathlib import Path

from VAE_Anime_Encoder import VAE_Encoder
from VAE_Anime_Decoder import VAE_Decoder
from utils import setup_logging


class KLDivergenceLossLayer(tf.keras.layers.Layer):
    '''Give the KL Divergence Loss as a layer in the model.'''
    def __init__(self, **kwargs):
        super(KLDivergenceLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_loss = 1 + log_var - tf.square(mu) - tf.exp(log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        self.add_loss(kl_loss)
        return kl_loss


class VAE_Model():
    '''Create a full VAE model with encoder, decoder, and VAE network.'''
    def __init__(self, enc_input_shape=(64,64,3,), 
                latent_dim=512, output_dir="scratch_output"):
        self.enc_input_shape = enc_input_shape

        # Set the latent dimension and output directory
        self.latent_dim = latent_dim
        self.output_dir = Path(output_dir)

        # Initialize the encoder, decoder, and VAE
        self.init_encoder()
        self.init_decoder()
        self.init_VAE()


    def init_encoder(self):
        # Create a new VAE_Encoder object and set the encoder model
        self.encoder = VAE_Encoder(self.enc_input_shape, 
                                   self.latent_dim, self.output_dir)
        self.encoder.set_encoder_model()

    def init_decoder(self):
        # Create a new VAE_Decoder object and set the decoder model
        if self.encoder is None or self.encoder.encoder_net is None:
            logging.warning("The encoder must be initialized first.")
            return
        
        # The input shape of the encoder's encoder_flatten layer is 
        # the shape the decoder needs to recover from flattened data
        enc_output_shape = self.encoder.encoder_net.get_layer('encoder_flatten').input.shape
        self.decoder = VAE_Decoder(enc_output_shape, self.latent_dim, self.output_dir)
        self.decoder.set_decoder_model()

    def init_VAE(self):
        # Initialize the VAE model
        self.init_encoder()
        self.init_decoder()

        
        inputs = tf.keras.layers.Input(shape=self.enc_input_shape)

        # get mu, log_var, and z from the encoder output
        mu, log_var, z = self.encoder.encoder_net(inputs)
    
        # get reconstructed output from the decoder
        reconstruction = self.decoder.decoder_net(z)

        # define the inputs and outputs of the VAE
        self.vae_net = tf.keras.Model(inputs=inputs, 
                                      outputs=[reconstruction, mu, log_var],
                                      name="Full_VAE_Network")



    def show_model(self):
        # Display the model summary
        if self.vae_net:
            with open(self.output_dir / Path('model_summary.txt'), 'w') as f:
                # Redirect stdout to the file
                with redirect_stdout(f):
                    # Call model.summary(), which will now print to the file
                    self.vae_net.summary()
                    logging.info("Model summary has been saved to 'model_summary.txt'") 
                self.encoder.show_model()
                self.decoder.show_model()
        else:
            logging.info("Full VAE Model not yet defined")

      

if __name__ == '__main__':
    setup_logging()
    VAE = VAE_Model()
    VAE.init_VAE()
    VAE.show_model()

