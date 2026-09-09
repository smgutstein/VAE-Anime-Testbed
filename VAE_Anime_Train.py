import argparse
import logging
import numpy as np
import random

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
from VAE_Anime_RunArtifacts import write_run_summary
from VAE_Anime_ReferenceVAE import BestSSIMReferenceSaver
from VAE_Anime_Snapshotter import VAESnapshotter
from VAE_Anime_StepGuard import StepGuard, StepResult
from VAE_Anime_Training_Monitor import TrainingMonitor
from VAE_Anime_TrainStep import train_step, build_step_result
from VAE_Anime_Validate import evaluate_validation_set

from utils import setup_logging


def _set_all_seeds(seed: int, deterministic: bool = False):
    """Set Python, NumPy, and TensorFlow seeds.

    Args:
        seed: Integer seed value.
        deterministic: If True, request more deterministic TF behavior.
            This can reduce performance and is not guaranteed to make
            every GPU op bitwise identical.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as e:
            logging.warning(f"Could not enable TF op determinism: {e}")


class VAE_Trainer:
    """Class to train the VAE model on the anime faces dataset."""

    mse_loss = tf.keras.losses.MeanSquaredError()

    def __init__(self, config_file="config.ini"):
        # Load typed training params from config file
        self.cfg = TrainerConfig.from_file(config_file)

        # Make run randomness explicit and repeatable
        _set_all_seeds(self.cfg.seed, deterministic=self.cfg.deterministic)
        logging.info(
            f"Using random seed {self.cfg.seed} "
            f"(deterministic={self.cfg.deterministic})"
        )

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

        # Initialize datasets
        self.data = Datasets(
            self.output_dir,
            seed=self.cfg.seed,
            data_dir=self.cfg.data_dir,
            strict_reproducibility=self.cfg.deterministic,
        )
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

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.cfg.learning_rate
        )

        self.loss_policy = build_loss_policy(self.cfg)

        self.reference_saver = (
            BestSSIMReferenceSaver(self.output_dir, self.cfg)
            if self.cfg.save_ref_vae
            else None
        )

        # Trip-wire state
        self.prev_kl_tensor = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.beta_factor = tf.Variable(
            self.loss_policy.current_value(),
            dtype=tf.float32,
            trainable=False,
        )

        logging.info(f"Using loss policy: {self.loss_policy.policy_name}")

        self.monitor = TrainingMonitor(
            stats_dir=self.stats_dir,
            snapshot_every=self.cfg.snapshot_every,
        )

        self.step_guard = StepGuard(
            stats_dir=self.stats_dir,
            loss_policy=self.loss_policy,
            optimizer=self.optimizer,
            recent_good_maxlen=8,
            lr_backoff=self.cfg.tripwire_lr_backoff,
            lr_floor=self.cfg.tripwire_lr_floor,
            max_consecutive_tripwires=self.cfg.max_consecutive_tripwires,
            kl_jump_ratio_threshold=self.cfg.step_guard_kl_jump_ratio_threshold,
            kl_abs_threshold=self.cfg.step_guard_kl_abs_threshold,
            max_log_var_threshold=self.cfg.step_guard_max_log_var_threshold,
        )


    def train_loop(self):
        start_time = time()
        logging.info("Start Time: %s" % ctime())

        last_recon = None
        last_kl = None
        last_kl_weight = self.loss_policy.current_value()

        weight_update_events = [(0, 0, self.loss_policy.current_update_factor())]

        self.monitor.open()
        try:
            for epoch in range(self.cfg.epochs):
                logging.info("Start of epoch %d of %d at %s" % (epoch, self.cfg.epochs, ctime()))

                if (epoch + 1) % self.cfg.flush_every_epochs == 0:
                    self.monitor.flush()
                    logging.info("File Buffers Flushed")

                for step, x_batch_train in enumerate(self.data.training_dataset):
                    self.beta_factor.assign(self.loss_policy.current_value())
                    curr_kl_weight_before_update = float(self.loss_policy.current_value())

                    # Take gradient descent step
                    raw_step_output = train_step(
                        x_batch_train,
                        self.beta_factor,
                        self.prev_kl_tensor,
                        self,
                        self.mse_loss,
                        self.optimizer,
                    )

                    # Create tuple with all results of grad descent step
                    step_result = build_step_result(
                        raw_step_output=raw_step_output,
                        x_batch_train=x_batch_train,
                        epoch=epoch,
                        step=step,
                        kl_weight=curr_kl_weight_before_update,
                    )

                    # Decide if tripwire activated
                    decision = self.step_guard.handle_step(step_result)

                    if decision.should_raise:
                        raise decision.exception

                    if decision.should_continue:
                        continue

                    if decision.should_update_prev_kl:
                        self.prev_kl_tensor.assign(decision.prev_kl_value)

                    # Record recon & kl losses
                    curr_loss_recon = step_result.loss_recon
                    curr_loss_kl = step_result.loss_kl

                    last_recon = curr_loss_recon
                    last_kl = curr_loss_kl
                    last_kl_weight = curr_kl_weight_before_update

                    # Update according to loss policy
                    update_info = self.loss_policy.update(curr_loss_recon, curr_loss_kl)
                    weight_direction = update_info["weight_direction"]
                    num_maxes = update_info["num_maxes"]
                    test1 = update_info["test1"]
                    test2 = update_info["test2"]
                    max_kl_weight_seen = update_info["max_kl_weight_seen"]
                    min_kl_weight_seen = update_info["min_kl_weight_seen"]
                    curr_kl_weight = update_info["kl_weight"]

                    if update_info["update_factor_changed"]:
                        weight_update_events.append(
                            (epoch, step, update_info["update_factor"])
                        )

                    # Write current results/state to text file
                    # Record kl weight that was just used, not one to be used next
                    self.monitor.record_text_line(
                        epoch=epoch,
                        step=step,
                        loss_recon=curr_loss_recon,
                        loss_kl=curr_loss_kl,
                        curr_kl_weight=curr_kl_weight_before_update,
                        test1=test1,
                        test2=test2,
                        num_maxes=num_maxes,
                        max_kl_weight_seen=max_kl_weight_seen,
                        min_kl_weight_seen=min_kl_weight_seen,
                        window_len=update_info["window_len"],
                        kl_jump_ratio=step_result.kl_jump_ratio,
                        max_log_var=step_result.max_log_var,
                        min_log_var=step_result.min_log_var,
                        max_grad_norm=(
                            self.cfg.max_grad_norm
                            if self.cfg.max_grad_norm is not None
                            else step_result.grad_norm
                        ),
                    )

                    # Create image showing how validation images are reconstructed
                    # and how images are generated
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

                    # Append losses to running lists
                    # Record kl weight that was just used, not one to be used next
                    self.monitor.record_losses(
                        curr_loss_recon=curr_loss_recon,
                        curr_loss_kl=curr_loss_kl,
                        curr_kl_weight=curr_kl_weight_before_update,
                        recon_ssim=step_result.recon_ssim,
                        active_latent_dims=step_result.active_latent_dims,
                    )

                    # Add stats on latent layer to running lists
                    self.monitor.record_latent_stats(
                        mu_mean=tf.reduce_mean(step_result.mu, 0),
                        log_var_mean=tf.reduce_mean(step_result.log_var, 0),
                        mu_var=tf.math.reduce_variance(step_result.mu, 0),
                        log_var_var=tf.math.reduce_variance(step_result.log_var, 0),
                        kl_per_dim=step_result.kl_per_dim,
                    )

                    self.monitor.maybe_flush_step(step)

                    curr_time = time()
                    tot_delta_time = str(timedelta(seconds=curr_time - start_time))
                    out_str = f"Epoch: {epoch} of {self.cfg.epochs} step: {step} "
                    out_str += f"recon loss = {curr_loss_recon:.4f} "
                    out_str += f"kl_loss = {curr_loss_kl:.4e} "
                    out_str += f"{weight_direction} kl_weight = {curr_kl_weight_before_update:.4e} "
                    out_str += f"tot run time = {tot_delta_time}"
                    logging.info(out_str)

                # Held-out evaluation on the full validation split. Epoch-based
                # rather than step-based: is_snapshot_step fires several times
                # per epoch, which would cost more than training.
                is_last_epoch = epoch == self.cfg.epochs - 1
                if (epoch % self.cfg.validate_every_epochs == 0) or is_last_epoch:
                    val_result = evaluate_validation_set(
                        validation_dataset=self.data.validation_dataset,
                        encoder_net=self.vae.encoder.encoder_net,
                        decoder_net=self.vae.decoder.decoder_net,
                        loss_fn=self.mse_loss,
                        num_input_pixels=self.vae.encoder.num_input_pixels,
                        active_dim_kl_threshold=self.cfg.active_dim_kl_threshold,
                        seed=self.cfg.seed,
                    )
                    validation_kl_weight = float(self.loss_policy.current_value())
                    self.monitor.record_validation(
                        epoch=epoch,
                        step=step,
                        kl_weight=validation_kl_weight,
                        result=val_result,
                    )
                    if self.reference_saver is not None:
                        self.reference_saver.consider(
                            vae=self.vae,
                            val_result=val_result,
                            epoch=epoch,
                            step=step,
                            kl_weight=validation_kl_weight,
                        )
                    logging.info(
                        "Validation epoch %d: recon_mu=%.4f recon_sampled=%.4f "
                        "kl=%.4e ssim_mu=%.4f active_dims=%d "
                        "agg_post_var mean/min/max=%.3f/%.3f/%.3f over %d images",
                        epoch,
                        val_result.recon_mu,
                        val_result.recon_sampled,
                        val_result.kl_loss,
                        val_result.ssim_mu,
                        val_result.active_latent_dims,
                        val_result.agg_post_var_mean,
                        val_result.agg_post_var_min,
                        val_result.agg_post_var_max,
                        val_result.n_images,
                    )

            logging.info("End Time %s" % ctime())
            delta_time = str(timedelta(seconds=time() - start_time))
            logging.info("Running Time %s", delta_time)

            if self.cfg.save_net:
                model_path = self.stats_dir / Path("anime.keras")
                logging.info(f"Saving the model to {model_path}")
                self.vae.vae_net.save(model_path)
            else:
                logging.info("Model not saved")

            logging.info(f"Number of kl_weight changes: {len(weight_update_events)}")
            logging.info(weight_update_events)

            guard_report = (
                self.step_guard.tripwire_report()
                if getattr(self, "step_guard", None) is not None
                else None
            )
            write_run_summary(
                output_dir=self.output_dir,
                cfg=self.cfg,
                loss_policy=self.loss_policy,
                status="completed",
                start_time=start_time,
                end_time=time(),
                final_recon=last_recon,
                final_kl=last_kl,
                final_kl_weight=last_kl_weight,
                tripwire_report=guard_report,
            )

        except Exception as e:
            guard_report = (
                self.step_guard.tripwire_report()
                if getattr(self, "step_guard", None) is not None
                else None
            )
            write_run_summary(
                output_dir=self.output_dir,
                cfg=self.cfg,
                loss_policy=self.loss_policy,
                status="failed",
                start_time=start_time,
                end_time=time(),
                final_recon=last_recon,
                final_kl=last_kl,
                final_kl_weight=last_kl_weight,
                tripwire_report=guard_report,
                error_message=str(e),
            )
            raise

        finally:
            self.monitor.close()

#############################################################

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the anime VAE from a config file."
    )
    parser.add_argument(
        '-c', '--config_file',
        type=str,
        default='config.ini',
        help='Config file path or config filename inside ./configs'
    )
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
        ar.movie_builder.make_images_movie()
        ar.make_pareto_curve_graph()
        ar.make_pareto_movie()
        if vae.cfg.make_mu_log_var_movies:
            ar.movie_builder.make_mu_log_var_movie(log_var_graph=True)
            ar.movie_builder.make_mu_log_var_movie(log_var_graph=False)

    return 0
#############################################################
   

if __name__ == "__main__":
    raise SystemExit(main())
