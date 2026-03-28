import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shutil
import sys

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide INFO, WARNING, and most ERROR logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # removes the oneDNN startup note
import tensorflow as tf

from collections import deque
from datetime import timedelta
from pathlib import Path
from time import time, ctime

from VAE_Anime_Analysis import AnalyzeResults
from VAE_Anime_Config import TrainerConfig
from VAE_Anime_Datasets import Datasets
from VAE_Anime_Full_Model import VAE_Model


from utils import DeltaGenerator
from utils import delt_mul, delt_div
from utils import get_git_hash
from utils import get_next_experiment_dir
from utils import load_and_validate_config
from utils import set_all_seeds
from utils import setup_logging
from utils import sign


class VAE_Trainer:
    '''Class to train the VAE model on the anime faces dataset'''

    # Initialize the loss metrics
    loss_metric_recon = tf.keras.metrics.Mean()
    loss_metric_kl = tf.keras.metrics.Mean()
    mse_loss = tf.keras.losses.MeanSquaredError()

    def __init__(self, config_file="config.ini"):
        # Load the config file
        assert Path(config_file).exists(), "Config file does not exist"
        
        # Load typed training params from config file
        self.cfg = TrainerConfig.from_file(config_file)
        self.config_file = str(self.cfg.config_file)

        # Make run randomness explicit and repeatable
        set_all_seeds(self.cfg.seed, deterministic=self.cfg.deterministic)
        logging.info(f"Using random seed {self.cfg.seed} (deterministic={self.cfg.deterministic})")

        # Set the output directory for this experiment
        self.curr_expt, self.output_dir = get_next_experiment_dir(self.cfg.parent_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(self.config_file, self.output_dir / "config.ini")
        logging.info(f"Storing Expt {self.curr_expt} in {self.output_dir}")


        # Record the git hash used for this run, along with
        # the current branch, last commit comment & uncommitted changes
        # in the Notes.txt file in the output directory
        with open(self.output_dir / Path("Notes.txt"), 'w') as f:
            hash_str = get_git_hash()
            f.write("Git Hash: \n")
            f.write(hash_str)
            f.write("\n")   
            f.write(f"Random Seed: {self.cfg.seed}\n")
            f.write(f"Deterministic TF Ops: {self.cfg.deterministic}\n")

        # Set up the output directories
        self.raw_image_dir = self.output_dir / "raw_images"
        self.raw_image_dir.mkdir(parents=True, exist_ok=True)

        self.stats_dir = self.output_dir / "stats"  
        self.stats_dir.mkdir(parents=True, exist_ok=True)   

        self.movies_dir = self.output_dir / "movies"
        self.movies_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the VAE model
        self.model_info_dir = self.output_dir / "model_info"
        self.model_info_dir.mkdir(parents=True, exist_ok=True)
        self.vae = VAE_Model(
            enc_input_shape=(self.cfg.image_size, self.cfg.image_size, 3),
            latent_dim=self.cfg.latent_dim,
            base_filters=self.cfg.base_filters,
            filter_factors=self.cfg.filter_factors,
            encode_dense_units=self.cfg.encode_dense_units,
            kernel_size=self.cfg.kernel_size,
            output_dir=self.model_info_dir,
        )

        # Initialize the Datasets class
        self.data = Datasets(self.output_dir, seed=self.cfg.seed, 
                             data_dir=self.cfg.data_dir)
        self.data.set_data_params(
            batch_size=self.cfg.batch_size,
            image_size=self.cfg.image_size,
            val_split=self.cfg.val_split,
            shuffle_buffer=self.cfg.shuffle_buffer,
            train_drop_remainder=self.cfg.train_drop_remainder,
        )
        self.data.download_data()
        self.data.make_train_and_validation_sets()

        # Constant idxs of test images
        self.fixed_test_img_idxs = None
        self.fixed_gen_img_seeds = tf.random.normal(shape=[4, self.vae.latent_dim])

        # Training parameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)
        self.kl_adj_factor = self.cfg.kl_adj_factor
        self.kl_adj_factor_max = self.cfg.kl_adj_factor_max
        self.kl_adj_update_factor = self.cfg.kl_adj_update_factor
        self.kl_adj_factor_queue = deque(maxlen=self.cfg.running_window)
        self.kl_adj_factor_delta_queue = deque(maxlen=self.cfg.running_window)



        # Initialize Delta Generator 
        #   Creates increment & decrement functions 
        #   that multiply/divide by  1+self.kl_adj_update_factor
        self.delta_gen = DeltaGenerator(delt_mul, delt_div, self.kl_adj_update_factor)
        self.inc = self.delta_gen.inc_func 
        self.dec = self.delta_gen.dec_func

        self.kl_adj_tensor = tf.Variable(self.kl_adj_factor, dtype=tf.float32, trainable=False)

            

    def snapshot_vae_behavior (self, epoch=0, step=0, 
                               recon_loss=0, kl_loss=0):
        """ Takes a snapshot of the VAE's behavior at a given epoch and step,
            by creating a 4x8 grid of images. 
            
            The first row is the input images, the second row is 
            the VAE's reconstruction of those images,
            the third row is the average face created by the decoder,
            and the fourth row is a random face created by the decoder. 
            
            The first 4 columns are fixed, while the last 4 columns are random.
            Each image is saved to a file in the raw_images directory, from
            where a movie of the evolving behavior of the VAE can be created.
        """

        # Get 1 batch from validation set and convert
        # to list of numpy arrays
        test_dataset = self.data.validation_dataset.take(1)
        output_samples = next(iter(test_dataset)).numpy()

        batch_size = output_samples.shape[0]
        if batch_size == 0:
            raise RuntimeError("Validation batch is empty; cannot create VAE snapshot")

        # VAE's response to each member of test_dataset
        vae_predicted, _, _ = self.vae.vae_net.predict(output_samples)

        # Construct indices of images to be displayed
        # 4 indices are the same for each call to this procedure
        # 4 are rndly chosen each time
        fixed_count = min(4, batch_size)
        rnd_count   = min(4, batch_size)

        # initialize fixed indices once
        if self.fixed_test_img_idxs is None or len(self.fixed_test_img_idxs) != fixed_count:
            self.fixed_test_img_idxs = np.random.choice(batch_size,
                                                        size=fixed_count,
                                                        replace=False)

        # random indices each snapshot
        rnd_test_img_idxs = np.random.choice(batch_size,
                                             size=rnd_count,
                                             replace=False)

        test_img_idxs = np.concatenate([self.fixed_test_img_idxs,
                                        rnd_test_img_idxs], axis=0)        
        # Construct 8 zero-vector seeds to show 'average' face
        # created by decoder
        zero_vector = tf.zeros(shape=[8, self.vae.latent_dim])
        avg_images = self.vae.decoder.decoder_net.predict(zero_vector)  

        # Construct 8 seeds of images to be generated by decoder
        # 4 seeds are the same for each call to this procedure
        # 4 are rndly chosen each time
        rnd_gen_img_seeds = tf.random.normal(shape=[4, self.vae.latent_dim])
        gen_img_seeds = tf.concat([self.fixed_gen_img_seeds, 
                                   rnd_gen_img_seeds], axis=0)
        gen_images = self.vae.decoder.decoder_net.predict(gen_img_seeds)
        
        num_idxs = len(test_img_idxs)
        fig = plt.figure(figsize=(8,5))
        for ctr,idx in enumerate(test_img_idxs):
            # Display the input images
            plt.subplot(4, num_idxs, ctr+1)
            img1 = output_samples[idx, :, :, :] * 255
            img1 = img1.astype('int32')
            plt.axis('off')
            plt.imshow(img1)
            
            # Display the VAE's reconstruction of the input images
            plt.subplot(4, num_idxs, ctr+1+num_idxs)
            img2 = vae_predicted[idx, :, :, :] * 255
            img2 = img2.astype('int32')
            plt.axis('off')
            plt.imshow(img2)

            # Display the average face created by the decoder
            plt.subplot(4, num_idxs, ctr+1+2*num_idxs)
            img3 = avg_images[ctr, :, :, :] * 255
            img3 = img3.astype('int32')
            plt.axis('off')
            plt.imshow(img3)

            # Display a random face created by the decoder
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
        plt.close()

    
    @staticmethod
    @tf.function(reduce_retracing=True)
    def train_step(x_batch_train, kl_adj_tensor, vae_obj, loss_fn, optimizer):
        model = vae_obj.vae_net

        with tf.GradientTape() as tape:
            reconstructed, mu, log_var = model(x_batch_train)

            # compute reconstruction loss
            loss_recon = loss_fn(x_batch_train, reconstructed) * vae_obj.encoder.num_input_pixels

            # get KLD regularization loss 
            loss_kl = tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var)) * -0.5

            # Compute weighted total loss
            loss_tot = loss_recon + kl_adj_tensor * loss_kl

        grads = tape.gradient(loss_tot, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_recon, loss_kl, mu, log_var, grads



    ###########################################################
    def train_loop(self, running_window=20):

        # Set Timing Parameters
        start_time = time()
        logging.info("Start Time: %s" % (ctime()))

        # Temp variable to see if I really do adjust the update factor
        adj_ctr = [(0, 0, self.kl_adj_update_factor)]

        # Initialize performance trackers
        prev_loss_recon = np.inf
        prev_loss_kl = np.inf

        # Lists of values for later plottting & diagnostics
        recon_loss_list = []
        kl_loss_list = []
        adj_kl_factor_list = []
        grad_list = []
        mu_list = []
        log_var_list = []
        mu_list2 = []
        log_var_list2 = []

        with (
            open(self.stats_dir / Path("losses_file.txt"), 'w') as loss_file,
            open(self.stats_dir / Path("loss_lists.pkl"), "wb") as f1,
            open(self.stats_dir / Path("mu_log_var_lists.pkl"), "wb") as f2,
            open(self.stats_dir / Path("mu_log_var_lists2.pkl"), "wb") as f3
        ):
            loss_file.write(f"Epoch -- Step -- Recon Loss -- KL Loss     -- KL_Adj_Factor\n")

            for epoch in range(self.cfg.epochs):
                logging.info('Start of epoch %d at %s' % (epoch, ctime()))

                # Flush buffers
                if (epoch + 1) % 100 == 0:
                    loss_file.flush()
                    f1.flush()
                    f2.flush()
                    f3.flush()
                    logging.info("File Buffers Flushed ")

                # Iterate over the batches of the dataset.
                for step, x_batch_train in enumerate(self.data.training_dataset):
                    # Convert kl_adj_factor to tensor for tf.function
                    self.kl_adj_tensor.assign(self.kl_adj_factor)

                    # Call static train_step
                    loss_recon, loss_kl, mu, log_var, grads = self.train_step(
                        x_batch_train,
                        self.kl_adj_tensor,
                        self.vae, 
                        self.mse_loss,
                        self.optimizer)
                    
                    # Get Current Losses
                    curr_loss_recon = loss_recon.numpy()
                    curr_loss_kl = loss_kl.numpy()

                    if np.isnan(curr_loss_recon) or np.isnan(curr_loss_kl):
                        logging.error("Nan in loss")
                        import pdb
                        pdb.set_trace()
                    elif np.isinf(curr_loss_recon) or np.isinf(curr_loss_kl):
                        logging.error("Inf in loss")
                        import pdb
                        pdb.set_trace()
 
                    # KL balancing
                    if curr_loss_recon >= prev_loss_recon:
                        # Recon loss is getting worse, decrease emphasis on KL Loss
                        self.kl_adj_factor = self.dec(self.kl_adj_factor)
                        adj_str = "-"
                    else:
                        # If recon loss improves, but kl didn't
                        self.kl_adj_factor = self.inc(self.kl_adj_factor)
                        adj_str = "+"

                    # Cap KL Loss Factor - This is tragically arbitrary
                    self.kl_adj_factor = min(self.kl_adj_factor, self.kl_adj_factor_max)

                    self.kl_adj_factor_queue.append(self.kl_adj_factor)
                    if len(self.kl_adj_factor_queue) >= 2:
                        delta = sign(self.kl_adj_factor_queue[-1] - self.kl_adj_factor_queue[-2])
                        self.kl_adj_factor_delta_queue.append(delta)

                    # Check if kl_adj_factor is bouncing too much at top of range
                    # If so, decrease the update factor
                    num_maxes = sum([1 for x in self.kl_adj_factor_queue if x == self.kl_adj_factor_max])
                    test1 = num_maxes > 0.4 * len(self.kl_adj_factor_queue)
                    test2 = num_maxes < 0.6 * len(self.kl_adj_factor_queue)

                    if test1 and test2:
                        self.kl_adj_update_factor *= 0.9
                        self.delta_gen = DeltaGenerator(delt_mul, delt_div, self.kl_adj_update_factor)
                        self.inc = self.delta_gen.inc_func
                        self.dec = self.delta_gen.dec_func
                        self.kl_adj_factor_queue.clear()
                        self.kl_adj_factor_queue.append(self.kl_adj_factor)
                        self.kl_adj_factor_delta_queue.clear()
                        adj_ctr.append((epoch, step, self.kl_adj_update_factor))

                    prev_loss_recon = curr_loss_recon
                    prev_loss_kl = curr_loss_kl

                    # Calculate Total Effective Loss
                    max_kl_adj_factor = max(self.kl_adj_factor_queue)
                    min_kl_adj_factor  = min(self.kl_adj_factor_queue)
                    loss_file.write(f"{epoch} -- {step} -- {loss_recon:.4f} -- {loss_kl:.4e} -- {self.kl_adj_factor:.4e}  ")
                    loss_file.write(f"{test1} {test2} {num_maxes} ")
                    loss_file.write(f"{max_kl_adj_factor } {min_kl_adj_factor } {len(self.kl_adj_factor_queue)}\n")

                    # Logging + metrics
                    if step % self.cfg.snapshot_every == 0:
                        self.snapshot_vae_behavior(epoch, step, curr_loss_recon, curr_loss_kl)

                    recon_loss_list.append(curr_loss_recon)
                    kl_loss_list.append(curr_loss_kl)
                    adj_kl_factor_list.append(self.kl_adj_factor)

                    gl_mags = [np.max(np.abs(x.numpy())) for x in grads]
                    grad_list.append(gl_mags)

                    # Track means of mu and log_var
                    mu_list.append(tf.reduce_mean(mu, 0))
                    log_var_list.append(tf.reduce_mean(log_var, 0))

                    # Track variances of mu and log_var
                    mu_list2.append(tf.math.reduce_variance(mu, 0))
                    log_var_list2.append(tf.math.reduce_variance(log_var, 0))

                    self.loss_metric_recon(loss_recon)
                    self.loss_metric_kl(loss_kl)

                    if step % self.cfg.snapshot_every == 0:
                        pickle.dump([recon_loss_list, kl_loss_list, adj_kl_factor_list], f1)
                        pickle.dump([mu_list, log_var_list], f2)
                        pickle.dump([mu_list2, log_var_list2], f3)

                        recon_loss_list.clear()
                        kl_loss_list.clear()
                        adj_kl_factor_list.clear()
                        mu_list.clear()
                        log_var_list.clear()
                        mu_list2.clear()
                        log_var_list2.clear()

                    curr_time = time()
                    tot_delta_time = str(timedelta(seconds=curr_time - start_time))
                    out_str = f"Epoch: {epoch} step: {step} "
                    out_str += f"recon loss = {curr_loss_recon:.4f} "
                    out_str += f"kl_loss = {curr_loss_kl:.4e} "
                    out_str += f"{adj_str} kl_adj_factor = {self.kl_adj_factor:.4e} "
                    out_str += f"tot run time = {tot_delta_time}"
                    logging.info(out_str)

            if recon_loss_list or kl_loss_list or adj_kl_factor_list:
                pickle.dump([recon_loss_list, kl_loss_list, adj_kl_factor_list], f1)
                pickle.dump([mu_list, log_var_list], f2)
                pickle.dump([mu_list2, log_var_list2], f3)
                loss_file.flush()
                f1.flush()
                f2.flush()
                f3.flush()
                logging.info("Flushed final partial stats buffers")

            logging.info("End Time %s" % (ctime()))
            delta_time = str(timedelta(seconds=time() - start_time))
            logging.info("Running Time %s", (delta_time))

            if self.cfg.save_net:
                logging.info(f"Saving the model to {self.stats_dir / Path('anime.keras')}")
                self.vae.vae_net.save(self.stats_dir / Path("anime.keras"))
            else:
                logging.info("Model not saved")

            logging.info(f"Number of kl_adj_factor changes: {len(adj_ctr)}")
            logging.info(adj_ctr)


    #############################################################

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set some params for training & output dir.')
    parser.add_argument('-c', '--config_file', type=str, nargs='?',
                        default='config.ini', help='Config file')
    parser.add_argument("--log", default="INFO", help="Logging level")

    args = parser.parse_args()
    try:
        config = load_and_validate_config(args.config_file)
    except Exception as e:
        logging.critical(f"Invalid config file: {e}")
        sys.exit(1)

    setup_logging(args.log)

    vae = VAE_Trainer(args.config_file)
    vae.vae.show_model()
    vae.data.display_sample_data('t', vae.cfg.train_preview_count)
    vae.data.display_sample_data('v', vae.cfg.valid_preview_count)
    if vae.cfg.take_initial_snapshot:
        vae.snapshot_vae_behavior()
    vae.train_loop()

    if vae.cfg.run_analysis:
        logging.info("Starting to analyze results....")
        ar = AnalyzeResults(args.config_file, vae.curr_expt)
        ar.make_singleton_graphs()
        ar.make_images_movie()
        ar.make_pareto_curve_graph()
        ar.make_paretoish_movie()
        if vae.cfg.make_mu_log_var_movies:
            ar.make_mu_log_var_movie(log_var_graph=True)
            ar.make_mu_log_var_movie(log_var_graph=False)
