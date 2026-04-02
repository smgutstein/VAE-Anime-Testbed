import argparse
import json
import logging
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
from VAE_Anime_LossPolicy import build_loss_policy
from VAE_Anime_Snapshotter import VAESnapshotter
from VAE_Anime_Training_Monitor import TrainingMonitor

from utils import set_all_seeds
from utils import setup_logging

class TrainingDivergedError(RuntimeError):
    pass

class VAE_Trainer:
    '''Class to train the VAE model on the anime faces dataset'''

    # Initialize the loss metrics
    mse_loss = tf.keras.losses.MeanSquaredError()

    def __init__(self, config_file="config.ini"):
        
        # Load typed training params from config file
        self.cfg = TrainerConfig.from_file(config_file)

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

        self.snapshotter = VAESnapshotter(
            raw_image_dir=self.raw_image_dir,
            latent_dim=self.vae.latent_dim,
        )
        # Training parameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)

        self.loss_policy = build_loss_policy(self.cfg)
        self.kl_weight_tensor = tf.Variable(
            self.loss_policy.current_value(),
            dtype=tf.float32,
            trainable=False,
        )

        logging.info(f"Using loss policy: {self.loss_policy.policy_name}")


        self.monitor = TrainingMonitor(
            stats_dir=self.stats_dir,
            snapshot_every=self.cfg.snapshot_every,
        )       

    def save_failure_tensors(self, epoch, step, x_batch_train,
                             mu, log_var, curr_loss_recon, curr_loss_kl,
                             diagnostics=None, max_items=32):
        sigma = tf.exp(0.5 * log_var).numpy()

        x_np = x_batch_train.numpy()[:max_items]
        mu_full = mu.numpy()
        log_var_full = log_var.numpy()

        mu_np = mu_full[:max_items]
        log_var_np = log_var_full[:max_items]
        sigma_np = sigma[:max_items]
        flat_idx = np.argsort(log_var_full.ravel())[-20:]

        out_path = self.stats_dir / Path(f"failure_tensors_epoch{epoch}_step{step}.npz")


        payload = {
            "epoch": np.array(epoch, dtype=np.int32),
            "step": np.array(step, dtype=np.int32),
            "loss_recon": np.array(curr_loss_recon, dtype=np.float32),
            "loss_kl": np.array(curr_loss_kl, dtype=np.float32),
            "kl_weight": np.array(self.loss_policy.current_value(), dtype=np.float32),
            "x_batch_train": x_np,
            "mu": mu_np,
            "log_var": log_var_np,
            "sigma": sigma_np,
            "top20_log_var_values": log_var_full.ravel()[flat_idx],
            "top20_sigma_values": sigma.ravel()[flat_idx],
            "top20_flat_indices": flat_idx,        
            }

        if diagnostics is not None:
            payload["diagnostics_text"] = np.array(repr(diagnostics), dtype=object)

        np.savez_compressed(out_path, **payload)
        logging.error("Saved failure tensors to %s", out_path)
        
    @staticmethod
    @tf.function(reduce_retracing=True)
    def train_step(x_batch_train, kl_weight_tensor, vae_obj, loss_fn, optimizer):
        model = vae_obj.vae_net

        with tf.GradientTape() as tape:
            reconstructed, mu, log_var = model(x_batch_train)

            # compute reconstruction loss
            loss_recon = loss_fn(x_batch_train, reconstructed) * vae_obj.encoder.num_input_pixels

            # get KLD regularization loss 
            loss_kl = tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var)) * -0.5

            # Compute weighted total loss
            loss_tot = loss_recon + kl_weight_tensor * loss_kl

        grads = tape.gradient(loss_tot, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_recon, loss_kl, mu, log_var



    ###########################################################
    def train_loop(self):

        # Set Timing Parameters
        start_time = time()
        logging.info("Start Time: %s" % (ctime()))
        last_recon = None
        last_kl = None
        last_kl_weight = self.loss_policy.current_value()

        # Track moments when the adaptive loss-policy shrinks its update factor
        weight_update_events = [(0, 0, self.loss_policy.current_update_factor())]

        self.monitor.open()
        try:

            for epoch in range(self.cfg.epochs):
                logging.info('Start of epoch %d at %s' % (epoch, ctime()))

                # Flush buffers
                if (epoch + 1) % 100 == 0:
                    self.monitor.flush()
                    logging.info("File Buffers Flushed")

                # Iterate over the batches of the dataset.
                for step, x_batch_train in enumerate(self.data.training_dataset):

                    # Convert current KL factor to tensor for tf.function
                    self.kl_weight_tensor.assign(self.loss_policy.current_value())

                    # Call static train_step
                    loss_recon, loss_kl, mu, log_var = self.train_step(
                        x_batch_train,
                        self.kl_weight_tensor,
                        self.vae, 
                        self.mse_loss,
                        self.optimizer)
                    
                    # Get Current Losses
                    curr_loss_recon = loss_recon.numpy()
                    curr_loss_kl = loss_kl.numpy()

                    last_recon = curr_loss_recon
                    last_kl = curr_loss_kl
                    last_kl_weight = self.loss_policy.current_value()

                    mu_finite = bool(tf.reduce_all(tf.math.is_finite(mu)).numpy())
                    log_var_finite = bool(tf.reduce_all(tf.math.is_finite(log_var)).numpy())

                    loss_bad = (
                        np.isnan(curr_loss_recon) or np.isnan(curr_loss_kl) or
                        np.isinf(curr_loss_recon) or np.isinf(curr_loss_kl)
                    )
                    latent_bad = (not mu_finite) or (not log_var_finite)

                    if loss_bad or latent_bad:
                        diagnostics = latent_diagnostics(mu, log_var)

                        logging.error(
                            "Training diverged at epoch=%d step=%d recon=%s kl=%s kl_weight=%s diagnostics=%s",
                            epoch,
                            step,
                            curr_loss_recon,
                            curr_loss_kl,
                            self.loss_policy.current_value(),
                            diagnostics,
                        )

                        self.save_failure_tensors(
                            epoch=epoch,
                            step=step,
                            x_batch_train=x_batch_train,
                            mu=mu,
                            log_var=log_var,
                            curr_loss_recon=curr_loss_recon,
                            curr_loss_kl=curr_loss_kl,
                            diagnostics=diagnostics,
                        )

                        raise TrainingDivergedError(
                            f"Training diverged at epoch={epoch}, step={step}, "
                            f"recon={curr_loss_recon}, kl={curr_loss_kl}, "
                            f"kl_weight={self.loss_policy.current_value()}"
                        )
 
                    update_info = self.loss_policy.update(curr_loss_recon, curr_loss_kl)
                    weight_direction = update_info["weight_direction"]
                    num_maxes = update_info["num_maxes"]
                    test1 = update_info["test1"]
                    test2 = update_info["test2"]
                    max_kl_weight_seen = update_info["max_kl_weight_seen"]
                    min_kl_weight_seen = update_info["min_kl_weight_seen"]
                    curr_kl_weight = update_info["kl_weight"]

                    if update_info["update_factor_changed"]:
                        weight_update_events.append((epoch, step, update_info["update_factor"]))

                    
                    self.monitor.record_text_line(
                        epoch=epoch,
                        step=step,
                        loss_recon=loss_recon,
                        loss_kl=loss_kl,
                        curr_kl_weight=curr_kl_weight,
                        test1=test1,
                        test2=test2,
                        num_maxes=num_maxes,
                        max_kl_weight_seen=max_kl_weight_seen,
                        min_kl_weight_seen=min_kl_weight_seen,
                        window_len=update_info["window_len"],
                    )

                    # Logging + metrics
                    if self.monitor.is_snapshot_step(step):
                        self.snapshotter.save_snapshot(
                            validation_dataset=self.data.validation_dataset,
                            vae_net=self.vae.vae_net,
                            decoder_net=self.vae.decoder.decoder_net,
                            epoch=epoch,
                            step=step,
                            recon_loss=curr_loss_recon,
                            kl_loss=curr_loss_kl,
                        )
                    self.monitor.record_losses(
                        curr_loss_recon=curr_loss_recon,
                        curr_loss_kl=curr_loss_kl,
                        curr_kl_weight=curr_kl_weight,
                    )

                    self.monitor.record_latent_stats(
                        mu_mean=tf.reduce_mean(mu, 0),
                        log_var_mean=tf.reduce_mean(log_var, 0),
                        mu_var=tf.math.reduce_variance(mu, 0),
                        log_var_var=tf.math.reduce_variance(log_var, 0),
                    )


                    self.monitor.maybe_flush_step(step)

                    curr_time = time()
                    tot_delta_time = str(timedelta(seconds=curr_time - start_time))
                    out_str = f"Epoch: {epoch} step: {step} "
                    out_str += f"recon loss = {curr_loss_recon:.4f} "
                    out_str += f"kl_loss = {curr_loss_kl:.4e} "
                    out_str += f"{weight_direction} kl_weight = {curr_kl_weight:.4e} "
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

            logging.info(f"Number of kl_weight changes: {len(weight_update_events)}")
            logging.info(weight_update_events)

            self.write_run_summary(
                status="completed",
                start_time=start_time,
                end_time=time(),
                final_recon=last_recon,
                final_kl=last_kl,
                final_kl_weight=last_kl_weight,
            )

        except Exception as e:
            self.write_run_summary(
                status="failed",
                start_time=start_time,
                end_time=time(),
                final_recon=last_recon,
                final_kl=last_kl,
                final_kl_weight=last_kl_weight,
                error_message=str(e),
            )
            raise

        finally:
            self.monitor.close()

    #############################################################

    def write_run_summary(self, status, start_time, end_time,
                          final_recon=None, final_kl=None,
                          final_kl_weight=None, error_message=None,
                          ):
        
        summary_path = self.output_dir / "run_summary.json"

        runtime_seconds = None
        if start_time is not None and end_time is not None:
            runtime_seconds = float(end_time - start_time)

        summary = {
            "experiment_dir": str(self.output_dir),
            "experiment_name": self.cfg.expt_name,
            "status": status,
            "error": error_message,

            "config_file": str(self.cfg.config_file),
            "loss_policy": self.loss_policy.policy_name,
            "seed": self.cfg.seed,
            "deterministic": self.cfg.deterministic,

            "epochs": self.cfg.epochs,
            "learning_rate": self.cfg.learning_rate,
            "latent_dim": self.cfg.latent_dim,
            "batch_size": self.cfg.batch_size,
            "image_size": self.cfg.image_size,
            "save_net": self.cfg.save_net,

            "beta": self.cfg.beta,
            "initial_kl_weight": self.cfg.kl_adj_factor,
            "max_kl_weight": self.cfg.kl_adj_factor_max,
            "kl_weight_update_factor": self.cfg.kl_adj_update_factor,

            "final_recon_loss": final_recon,
            "final_kl_loss": final_kl,
            "final_kl_weight": final_kl_weight,

            "start_time": ctime(start_time) if start_time is not None else None,
            "end_time": ctime(end_time) if end_time is not None else None,
            "runtime_seconds": runtime_seconds,
        }

        summary = {k: json_safe(v) for k, v in summary.items()}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info("Wrote run summary to %s", summary_path)

#############################################################

def json_safe(value):
    if value is None:
        return None
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def latent_diagnostics(mu, log_var):
    sigma = tf.exp(0.5 * log_var)

    def stats(x):
        x_np = x.numpy()
        finite = np.isfinite(x_np)
        finite_vals = x_np[finite]
        if finite_vals.size == 0:
            return {
                "finite_count": 0,
                "nonfinite_count": x_np.size,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        return {
            "finite_count": int(finite.sum()),
            "nonfinite_count": int((~finite).sum()),
            "min": float(finite_vals.min()),
            "max": float(finite_vals.max()),
            "mean": float(finite_vals.mean()),
            "std": float(finite_vals.std()),
        }

    return {
        "mu": stats(mu),
        "log_var": stats(log_var),
        "sigma": stats(sigma),
        "largest_log_var": np.sort(log_var.numpy().ravel())[-10:].tolist(),
        "largest_sigma": np.sort(sigma.numpy().ravel())[-10:].tolist(),
    }    

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
        vae.snapshotter.save_snapshot(
            validation_dataset=vae.data.validation_dataset,
            vae_net=vae.vae.vae_net,
            decoder_net=vae.vae.decoder.decoder_net,
        )    
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
