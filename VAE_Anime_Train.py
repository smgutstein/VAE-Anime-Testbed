import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shutil
import sys
import tensorflow as tf
import tensorflow_datasets as tfds

from collections import defaultdict
from datetime import timedelta
from IPython import display
from pathlib import Path
from time import time, ctime

from VAE_Anime_Datasets import Datasets
from VAE_Anime_Full_Model import VAE_Model
from utils import is_config_file
from utils import read_config_file
from utils import get_git_hash


class VAE_Trainer:

    loss_metric_recon = tf.keras.metrics.Mean()
    loss_metric_kl = tf.keras.metrics.Mean()
    mse_loss = tf.keras.losses.MeanSquaredError()
    bce_loss = tf.keras.losses.BinaryCrossentropy()

    def __init__(self, config_file="config.ini"):
        # Load the config file
        assert Path(config_file).exists(), "Config file does not exist"
        assert is_config_file(config_file), "Invalid config file"
        
        self.config_file = config_file
        self.load_config_file()

        # Set the output directories
        parent_output_dir = Path(self.parent_dir)
        parent_output_dir.mkdir(parents=True, exist_ok=True)
        num_expts = len([d for d in parent_output_dir.iterdir() 
                            if d.is_dir() and "original_images" not in str(d)])
        self.output_dir = parent_output_dir / f"expt_{num_expts+1}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.config_file, self.output_dir / Path("config.ini")) 


        # Record the git hash used for this run
        with open(self.output_dir / Path("Notes.txt"), 'w') as f:
            hash_str = get_git_hash()
            f.write("Git Hash: \n")
            f.write(hash_str)
            f.write("\n")   

        self.raw_image_dir = self.output_dir / "raw_images"
        self.raw_image_dir.mkdir(parents=True, exist_ok=True)

        self.stats_dir = self.output_dir / "stats"  
        self.stats_dir.mkdir(parents=True, exist_ok=True)   

        self.movies_dir = self.output_dir / "movies"
        self.movies_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the VAE model
        self.model_info_dir = self.output_dir / "model_info"
        self.model_info_dir.mkdir(parents=True, exist_ok=True)
        self.vae = VAE_Model(output_dir=self.model_info_dir)

        # Initialize the Datasets class
        self.data = Datasets(self.output_dir)
        self.data.set_data_params()
        self.data.download_data()
        self.data.make_train_and_validation_sets()
        self.data.validation_dataset

        # Initialize losses dictionary
        self.param_hist_dict = defaultdict(list)    

        # Constant idxs of test images
        self.fixed_test_img_idxs = np.random.choice(64, size=4)
        self.fixed_gen_img_seeds = tf.random.normal(shape=[4, self.vae.latent_dim])

        # Training parameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def load_config_file(self):
        config = read_config_file(self.config_file)
        
        self.epochs = int(config.get('Training_Parameters', 'epochs'))      
        self.learning_rate = float(config.get('Training_Parameters', 'learning_rate'))   
        self.kl_adj_factor = float(config.get('Training_Parameters', 'kl_adj_factor'))  
        self.kl_adj_factor_max = float(config.get('Training_Parameters', 'kl_adj_factor_max'))  
        self.parent_dir = config.get('Output_Parameters', 'parent_dir')
        temp = config.get('Output_Parameters', 'save_net').lower() 
        if temp == 'true' or temp == '1':
            self.save_net = True            
        elif temp == 'false' or temp == '0':
            self.save_net = False   
        else:
            print("Invalid value for save_net in config file. Expected True/False, true/false or 1/0")
            print(f"Found {temp} in config file. Will assume value of true")
            self.save_net = True


    def snapshot_vae_behavior (self, epoch=0, step=0, 
                               recon_loss=0, kl_loss=0):

        # Get 1 batch from validation set and convert
        # to list of numpy arrays
        test_dataset = self.data.validation_dataset.take(1)
        output_samples = []
        for input_image in tfds.as_numpy(test_dataset):
            output_samples = input_image

        # VAE's response to each member of test_dataset
        vae_predicted, _, _ = self.vae.vae_net.predict(test_dataset)

        # Construct indices of images to be displayed
        # 4 indices are the same for each call to this procedure
        # 4 are rndly chosen each time
        rnd_test_img_idxs = np.random.choice(64, size=4)
        test_img_idxs = np.concatenate([self.fixed_test_img_idxs, 
                                        rnd_test_img_idxs], axis=0)
        
        # Construct 8 zero-vector seeds to show 'average' face
        # created by decoder
        zero_vector = tf.zeros(shape=[8, self.vae.latent_dim])
        avg_images = self.vae.decoder.decoder_net.predict(zero_vector)  

        # Construct seeds of images to be generated by decoder
        # 4 seeds are the same for each call to this procedure
        # 4 are rndly chosen each time
        rnd_gen_img_seeds = tf.random.normal(shape=[4, self.vae.latent_dim])
        gen_img_seeds = tf.concat([self.fixed_gen_img_seeds, 
                               rnd_gen_img_seeds], axis=0)
        gen_images = self.vae.decoder.decoder_net.predict(gen_img_seeds)
        
        num_idxs = 8
        fig = plt.figure(figsize=(8,5))
        for ctr,idx in enumerate(test_img_idxs):
            plt.subplot(4, num_idxs, ctr+1)
            img1 = output_samples[idx, :, :, :] * 255
            img1 = img1.astype('int32')
            plt.axis('off')
            plt.imshow(img1)
            
            plt.subplot(4, num_idxs, ctr+1+num_idxs)
            img2 = vae_predicted[idx, :, :, :] * 255
            img2 = img2.astype('int32')
            plt.axis('off')
            plt.imshow(img2)

            plt.subplot(4, num_idxs, ctr+1+2*num_idxs)
            img3 = avg_images[ctr, :, :, :] * 255
            img3 = img3.astype('int32')
            plt.axis('off')
            plt.imshow(img3)

            plt.subplot(4, num_idxs, ctr+1+3*num_idxs)
            img4 = gen_images[ctr, :, :, :] * 255
            img4 = img4.astype('int32')
            plt.axis('off')
            plt.imshow(img4)

        # tight_layout minimizes the overlap between 2 sub-plots
        titleStr1 = "epoch: {}, step: {}".format(epoch, step)
        if recon_loss > 0:
            titleStr2 = "Recon Loss: {:.4f} KL Loss: {:.4f} ".format(recon_loss,kl_loss)
        else: 
            titleStr2 = "Recon Loss: N/A KL Loss: N/A"
        titleStr = titleStr1 + '\n' + titleStr2
        fig.suptitle(titleStr)
        
        file_name = f"image_at_epoch_{epoch:04d}_step{step:04d}.png"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.raw_image_dir / Path(file_name))

    def train_loop(self, running_window=20):

        # Set Timing Parameters
        start_time = time()
        last_time = start_time
        print("Start Time: ", ctime())

        # Initialize performance trackers
        prev_loss_recon = np.inf
        prev_loss_kl = np.inf

        recon_loss_list = []
        kl_loss_list = []
        adj_kl_factor_list = []
        grad_list = []
        mu_list=[]
        log_var_list=[]
        mu_list2=[]
        log_var_list2=[]

        with open(self.stats_dir / Path("losses_file.txt"), 'w') as loss_file:
            loss_file.write(f"Epoch -- Step -- Recon Loss -- KL Loss     -- KL_Adj_Factor\n")
            for epoch in range(self.epochs):
                print('Start of epoch %d at %s' % (epoch, ctime()))
    
                # Iterate over the batches of the dataset.
                for step, x_batch_train in enumerate(self.data.training_dataset):

                    with tf.GradientTape() as tape:
                        # feed a batch to the VAE model
                        reconstructed, mu, log_var = self.vae.vae_net(x_batch_train)

                        # compute reconstruction loss
                        loss_recon = self.mse_loss(x_batch_train, reconstructed) * self.vae.encoder.num_input_pixels # 64 * 64 * 3

                        # get KLD regularization loss 
                        loss_kl = self.vae.vae_net.losses[0]

                        # Get Current Losses
                        curr_loss_recon = loss_recon.numpy()
                        curr_loss_kl = loss_kl.numpy()

                        # Scale losses
                        if (curr_loss_recon >= prev_loss_recon) and (curr_loss_kl <= prev_loss_kl):
                            self.kl_adj_factor /= 2
                        elif (curr_loss_recon < prev_loss_recon):
                            self.kl_adj_factor *= 2
                        self.kl_adj_factor = min(self.kl_adj_factor, self.kl_adj_factor_max)
                        prev_loss_recon = curr_loss_recon
                        prev_loss_kl = curr_loss_kl

                        # Calculate Total Effective Loss
                        loss_file.write(f"{epoch}     --  {step}   --  {loss_recon:.4f} -- {loss_kl:.4e}  -- {self.kl_adj_factor:.4e}  \n")
                        loss_tot = loss_recon + self.kl_adj_factor*loss_kl
                        
 
                    # Get gradient of tital effective loss w/resp to trainable params
                    grads = tape.gradient(loss_tot, self.vae.vae_net.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.vae.vae_net.trainable_weights))

                    if step % 10 == 0:
                        display.clear_output(wait=False)    
                        self.snapshot_vae_behavior(epoch, step, 
                                                    loss_recon.numpy(), 
                                                    loss_kl.numpy())
 
                    recon_loss_list.append(loss_recon.numpy())
                    kl_loss_list.append(loss_kl.numpy())
                    adj_kl_factor_list.append(self.kl_adj_factor)

                    gl_mags = [np.max(np.abs(x.numpy())) for x in grads]
                    grad_list.append(gl_mags)

                    mu_list.append(tf.reduce_mean(mu,0))
                    log_var_list.append(tf.reduce_mean(log_var,0))
                    mu_list2.append(tf.math.reduce_variance(mu,0))
                    log_var_list2.append(tf.math.reduce_variance(log_var,0))

                    # compute the loss metric
                    # Remember:
                    #   loss_metric_recon = tf.keras.metrics.Mean()
                    #   loss_metric_kl = tf.keras.metrics.Mean()
                    self.loss_metric_recon(loss_recon)
                    self.loss_metric_kl(loss_kl)


                    if step % 10 == 0:
                        with open(self.stats_dir / Path("loss_lists.pkl"), "wb") as f:
                            pickle.dump([recon_loss_list, kl_loss_list, adj_kl_factor_list],f)
                        with open(self.stats_dir / Path("mu_log_var_lists.pkl"), "wb") as f:
                            pickle.dump([mu_list, log_var_list],f)
                        with open(self.stats_dir / Path("mu_log_var_lists2.pkl"), "wb") as f:
                            pickle.dump([mu_list2, log_var_list2],f)


                    curr_time = time()
                    #step_delta_time = str(timedelta(seconds = curr_time - last_time))
                    tot_delta_time = str(timedelta(seconds = curr_time - start_time))
                    last_time = curr_time

                    print('Epoch: %s step: %s mean loss = %s, kl_loss = %s, kl_adj_factor = %s, tot run time = %s' %
                        (epoch, step, loss_recon.numpy(), loss_kl.numpy(),
                            self.kl_adj_factor, tot_delta_time))
        print("End Time", ctime())
        delta_time = str(timedelta(seconds = curr_time - start_time))
        print("Running Time", delta_time)
        if self.save_net:
            print(f"Saving the model to {self.stats_dir / Path('anime.keras')}")
            self.vae.vae_net.save(self.stats_dir / Path("anime.keras"))
        else:
            print("Model not saved")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python VAE_Anime_Train.py <config_file>") 
        sys.exit(1)
    if not is_config_file(sys.argv[1]): 
        print("Invalid config file")
        sys.exit(1) 

    parser = argparse.ArgumentParser(description='Set some params for training & output dir.')
    parser.add_argument('config_file', type=str, 
                        default='config.ini', help='Config file')
    args = parser.parse_args()

    vae = VAE_Trainer(args.config_file)
    vae.vae.show_model()
    vae.data.display_sample_data('t', 25)
    vae.data.display_sample_data('v', 18)
    vae.snapshot_vae_behavior()
    vae.train_loop()
