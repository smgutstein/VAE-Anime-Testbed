import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide INFO, WARNING, and most ERROR logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # removes the oneDNN startup note
import tensorflow as tf

from datetime import timedelta
from pathlib import Path
from time import time, ctime

from VAE_Anime_Analysis import AnalyzeResults
from VAE_Anime_Config import TrainerConfig
from VAE_Anime_Datasets import Datasets
from VAE_Anime_ExperimentRun import ExperimentRun
from VAE_Anime_Full_Model import VAE_Model
from VAE_Anime_KL_Controller import KLController
from VAE_Anime_Training_Monitor import TrainingMonitor

from utils import set_all_seeds
from utils import setup_logging


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


        # Set up experiment/run artifacts
        self.run = ExperimentRun.create(self.cfg)
        self.curr_expt = self.run.expt_num
        self.output_dir = self.run.output_dir
        self.raw_image_dir = self.run.raw_image_dir
        self.stats_dir = self.run.stats_dir
        self.movies_dir = self.run.movies_dir
        self.model_info_dir = self.run.model_info_dir

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
        self.kl_controller = KLController(
            initial_factor=self.cfg.kl_adj_factor,
            max_factor=self.cfg.kl_adj_factor_max,
            update_factor=self.cfg.kl_adj_update_factor,
            running_window=self.cfg.running_window,
        )
        self.kl_adj_tensor = tf.Variable(
            self.kl_controller.current_value(),
            dtype=tf.float32,
            trainable=False,
        )

        self.monitor = TrainingMonitor(
            stats_dir=self.stats_dir,
            snapshot_every=self.cfg.snapshot_every,
        )       

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

        # Track moments when the controller shrinks its update factor
        adj_ctr = [(0, 0, self.kl_controller.current_update_factor())]
        grad_list = []

        self.monitor.open()
        try:

            for epoch in range(self.cfg.epochs):
                logging.info('Start of epoch %d at %s' % (epoch, ctime()))

                # Flush buffers
                if (epoch + 1) % 100 == 0:
                    self.monitor.loss_file.flush()
                    self.monitor.f_loss_lists.flush()
                    self.monitor.f_mu.flush()
                    self.monitor.f_var.flush()
                    logging.info("File Buffers Flushed ")

                # Iterate over the batches of the dataset.
                for step, x_batch_train in enumerate(self.data.training_dataset):

                    # Convert current KL factor to tensor for tf.function
                    self.kl_adj_tensor.assign(self.kl_controller.current_value())

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
 
                    update_info = self.kl_controller.update(curr_loss_recon, curr_loss_kl)
                    adj_str = update_info["adj_str"]
                    num_maxes = update_info["num_maxes"]
                    test1 = update_info["test1"]
                    test2 = update_info["test2"]
                    max_kl_adj_factor = update_info["max_factor_seen"]
                    min_kl_adj_factor = update_info["min_factor_seen"]
                    curr_kl_adj_factor = update_info["factor"]

                    if update_info["update_factor_changed"]:
                        adj_ctr.append((epoch, step, update_info["update_factor"]))

                    
                    self.monitor.record_text_line(
                        epoch=epoch,
                        step=step,
                        loss_recon=loss_recon,
                        loss_kl=loss_kl,
                        curr_kl_adj_factor=curr_kl_adj_factor,
                        test1=test1,
                        test2=test2,
                        num_maxes=num_maxes,
                        max_kl_adj_factor=max_kl_adj_factor,
                        min_kl_adj_factor=min_kl_adj_factor,
                        window_len=update_info["window_len"],
                    )

                    # Logging + metrics
                    if self.monitor.is_snapshot_step(step):
                        self.snapshot_vae_behavior(epoch, step, curr_loss_recon, curr_loss_kl)

                    self.monitor.record_losses(
                        curr_loss_recon=curr_loss_recon,
                        curr_loss_kl=curr_loss_kl,
                        curr_kl_adj_factor=curr_kl_adj_factor,
                    )

                    gl_mags = [np.max(np.abs(x.numpy())) for x in grads]
                    grad_list.append(gl_mags)

                    self.monitor.record_latent_stats(
                        mu_mean=tf.reduce_mean(mu, 0),
                        log_var_mean=tf.reduce_mean(log_var, 0),
                        mu_var=tf.math.reduce_variance(mu, 0),
                        log_var_var=tf.math.reduce_variance(log_var, 0),
                    )

                    self.loss_metric_recon(loss_recon)
                    self.loss_metric_kl(loss_kl)

                    self.monitor.maybe_flush_step(step)

                    curr_time = time()
                    tot_delta_time = str(timedelta(seconds=curr_time - start_time))
                    out_str = f"Epoch: {epoch} step: {step} "
                    out_str += f"recon loss = {curr_loss_recon:.4f} "
                    out_str += f"kl_loss = {curr_loss_kl:.4e} "
                    out_str += f"{adj_str} kl_adj_factor = {curr_kl_adj_factor:.4e} "
                    out_str += f"tot run time = {tot_delta_time}"
                    logging.info(out_str)


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

        finally:
            self.monitor.close()

    #############################################################

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set some params for training & output dir.')
    parser.add_argument('-c', '--config_file', type=str, nargs='?',
                        default='config.ini', help='Config file')
    parser.add_argument("--log", default="INFO", help="Logging level")

    args = parser.parse_args()
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
