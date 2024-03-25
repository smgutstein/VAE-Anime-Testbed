import numpy as np
import tensorflow as tf

from VAE_Anime_Encoder import VAE_Encoder
from VAE_Anime_Decoder import VAE_Decoder

def kl_reconstruction_loss(inputs, outputs, mu, log_sigma):
    """ Computes the Kullback-Leibler Divergence (KLD)
    Args:
        inputs -- batch from the dataset
        outputs -- output of the Sampling layer
        mu -- mean
        log_sigma -- log of standard deviation

    Returns:
        KLD loss
    """
    kl_loss = 1 + log_sigma - tf.square(mu) - tf.math.exp(log_sigma)
    return tf.reduce_mean(kl_loss) * -0.5

class VAE_Model():
    def __init__(self, enc_input_shape=(64,64,3,), 
                latent_dim=512, output_dir="scratch_output"):
        self.enc_input_shape = enc_input_shape
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        self.init_encoder()
        self.init_decoder()
        self.init_VAE()


    def init_encoder(self):
        self.encoder = VAE_Encoder(self.enc_input_shape, self.latent_dim)
        self.encoder.set_encoder_model()

    def init_decoder(self):
        if self.encoder is None or self.encoder.encoder_net is None:
            print("The encoder must be initialized first.")
            return
        enc_output_shape = self.encoder.encoder_net.get_layer('last_batch_normalization').input_shape
        self.decoder = VAE_Decoder(enc_output_shape, self.latent_dim)
        self.decoder.set_decoder_model()

    def init_VAE(self):
        self.init_encoder()
        self.init_decoder()

        
        inputs = tf.keras.layers.Input(shape=self.enc_input_shape)

        # get mu, sigma, and z from the encoder output
        mu, sigma, z = self.encoder.encoder_net(inputs)
    
        # get reconstructed output from the decoder
        reconstruction = self.decoder.decoder_net(z)

        # define the inputs and outputs of the VAE
        self.vae_net = tf.keras.Model(inputs=inputs, 
                                      outputs=[reconstruction, mu, sigma],
                                      name="Full_VAE_Network")
    
        # add the KL loss
        loss = kl_reconstruction_loss(inputs, z, mu, sigma)
        self.vae_net.add_loss(loss)

    def show_model(self):
        if self.vae_net:
            self.vae_net.summary()   
        else:
            print("Model not yet defined")

      

if __name__ == '__main__':
    VAE = VAE_Model()
    VAE.init_VAE()
    VAE.show_model()

