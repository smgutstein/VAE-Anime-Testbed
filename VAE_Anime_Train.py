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
from VAE_Anime_Snapshotter import VAESnapshotter
from VAE_Anime_StepGuard import (StepGuard, StepResult,)
from VAE_Anime_Training_Monitor import TrainingMonitor

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

        # Trip-wire state
        self.prev_kl_tensor = tf.Variable(1.0, dtype=tf.float32, trainable=False)
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

    @staticmethod
    @tf.function(reduce_retracing=True)
    def train_step(
        x_batch_train,
        kl_weight_tensor,
        prev_kl_tensor,
        vae_obj,
        loss_fn,
        optimizer,
    ):
        model = vae_obj.vae.vae_net
        num_input_pixels = vae_obj.vae.encoder.num_input_pixels

        with tf.GradientTape() as tape:
            reconstructed, mu, log_var = model(x_batch_train)

            loss_recon = loss_fn(x_batch_train, reconstructed) * num_input_pixels
            loss_kl = tf.reduce_mean(
                1.0 + log_var - tf.square(mu) - tf.exp(log_var)
            ) * -0.5
            loss_tot = loss_recon + kl_weight_tensor * loss_kl

        grads = tape.gradient(loss_tot, model.trainable_weights)

        non_none_grads = [g for g in grads if g is not None]
        grad_norm = (
            tf.linalg.global_norm(non_none_grads)
            if non_none_grads
            else tf.constant(0.0, dtype=tf.float32)
        )

        max_grad_norm = vae_obj.cfg.max_grad_norm
        kl_jump_ratio_threshold = vae_obj.cfg.step_guard_kl_jump_ratio_threshold
        kl_abs_threshold = vae_obj.cfg.step_guard_kl_abs_threshold
        max_log_var_threshold = vae_obj.cfg.step_guard_max_log_var_threshold

        if max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

        finite_losses = (
            tf.math.is_finite(loss_recon)
            & tf.math.is_finite(loss_kl)
            & tf.math.is_finite(loss_tot)
        )
        finite_mu = tf.reduce_all(tf.math.is_finite(mu))
        finite_log_var = tf.reduce_all(tf.math.is_finite(log_var))

        finite_grad_flags = [
            tf.reduce_all(tf.math.is_finite(g))
            for g in grads
            if g is not None
        ]
        finite_grads = (
            tf.reduce_all(tf.stack(finite_grad_flags))
            if finite_grad_flags
            else tf.constant(True)
        )

        max_log_var = tf.reduce_max(log_var)
        min_log_var = tf.reduce_min(log_var)
        max_abs_mu = tf.reduce_max(tf.abs(mu))

        prev_kl_safe = tf.maximum(
            prev_kl_tensor,
            tf.constant(1e-6, dtype=prev_kl_tensor.dtype),
        )
        kl_jump_ratio = loss_kl / prev_kl_safe

        kl_jump_bad = kl_jump_ratio > kl_jump_ratio_threshold
        kl_abs_bad = loss_kl > kl_abs_threshold
        log_var_bad = max_log_var > max_log_var_threshold

        def _reason_if(flag, name):
            return tf.cond(
                flag,
                lambda: tf.constant([name]),
                lambda: tf.constant([], dtype=tf.string),
            )

        toxic_reasons = tf.concat(
            [
                _reason_if(tf.logical_not(finite_losses), "nonfinite_losses"),
                _reason_if(tf.logical_not(finite_mu), "mu_nonfinite"),
                _reason_if(tf.logical_not(finite_log_var), "log_var_nonfinite"),
                _reason_if(tf.logical_not(finite_grads), "grad_nonfinite"),
                _reason_if(kl_jump_bad, "kl_jump_ratio"),
                _reason_if(kl_abs_bad, "kl_abs"),
                _reason_if(log_var_bad, "max_log_var"),
            ],
            axis=0,
        )

        toxic_step = (
            tf.logical_not(finite_losses)
            | tf.logical_not(finite_mu)
            | tf.logical_not(finite_log_var)
            | tf.logical_not(finite_grads)
            | kl_jump_bad
            | kl_abs_bad
            | log_var_bad
        )

        def do_apply():
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return tf.constant(True)

        def do_skip():
            return tf.constant(False)

        applied_update = tf.cond(toxic_step, do_skip, do_apply)

        return (
            loss_recon,
            loss_kl,
            mu,
            log_var,
            grad_norm,
            max_log_var,
            min_log_var,
            max_abs_mu,
            kl_jump_ratio,
            toxic_step,
            toxic_reasons,
            applied_update,
        )

    @staticmethod
    def _build_step_result(raw_step_output, x_batch_train, epoch, step, kl_weight):
        (
            loss_recon,
            loss_kl,
            mu,
            log_var,
            grad_norm,
            max_log_var,
            min_log_var,
            max_abs_mu,
            kl_jump_ratio,
            toxic_step,
            toxic_reasons,
            applied_update,
        ) = raw_step_output

        return StepResult(
            epoch=int(epoch),
            step=int(step),
            kl_weight=float(kl_weight),
            x_batch_train=x_batch_train,
            mu=mu,
            log_var=log_var,
            loss_recon=float(loss_recon.numpy()),
            loss_kl=float(loss_kl.numpy()),
            grad_norm=float(grad_norm.numpy()),
            max_log_var=float(max_log_var.numpy()),
            min_log_var=float(min_log_var.numpy()),
            max_abs_mu=float(max_abs_mu.numpy()),
            kl_jump_ratio=float(kl_jump_ratio.numpy()),
            toxic_step=bool(toxic_step.numpy()),
            toxic_reasons=tuple(x.decode("utf-8") for x in toxic_reasons.numpy().tolist()),
            applied_update=bool(applied_update.numpy()),
            mu_finite=bool(tf.reduce_all(tf.math.is_finite(mu)).numpy()),
            log_var_finite=bool(tf.reduce_all(tf.math.is_finite(log_var)).numpy()),
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
                    self.kl_weight_tensor.assign(self.loss_policy.current_value())
                    curr_kl_weight_before_update = float(self.loss_policy.current_value())

                    raw_step_output = self.train_step(
                        x_batch_train,
                        self.kl_weight_tensor,
                        self.prev_kl_tensor,
                        self,
                        self.mse_loss,
                        self.optimizer,
                    )

                    step_result = self._build_step_result(
                        raw_step_output=raw_step_output,
                        x_batch_train=x_batch_train,
                        epoch=epoch,
                        step=step,
                        kl_weight=curr_kl_weight_before_update,
                    )

                    decision = self.step_guard.handle_step(step_result)

                    if decision.should_raise:
                        raise decision.exception

                    if decision.should_continue:
                        continue

                    if decision.should_update_prev_kl:
                        self.prev_kl_tensor.assign(decision.prev_kl_value)

                    curr_loss_recon = step_result.loss_recon
                    curr_loss_kl = step_result.loss_kl

                    last_recon = curr_loss_recon
                    last_kl = curr_loss_kl
                    last_kl_weight = curr_kl_weight_before_update

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

                    self.monitor.record_text_line(
                        epoch=epoch,
                        step=step,
                        loss_recon=curr_loss_recon,
                        loss_kl=curr_loss_kl,
                        curr_kl_weight=curr_kl_weight,
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
                        mu_mean=tf.reduce_mean(step_result.mu, 0),
                        log_var_mean=tf.reduce_mean(step_result.log_var, 0),
                        mu_var=tf.math.reduce_variance(step_result.mu, 0),
                        log_var_var=tf.math.reduce_variance(step_result.log_var, 0),
                    )

                    self.monitor.maybe_flush_step(step)

                    curr_time = time()
                    tot_delta_time = str(timedelta(seconds=curr_time - start_time))
                    out_str = f"Epoch: {epoch} of {self.cfg.epochs} step: {step} "
                    out_str += f"recon loss = {curr_loss_recon:.4f} "
                    out_str += f"kl_loss = {curr_loss_kl:.4e} "
                    out_str += f"{weight_direction} kl_weight = {curr_kl_weight:.4e} "
                    out_str += f"tot run time = {tot_delta_time}"
                    logging.info(out_str)

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
            )

        except Exception as e:
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
                error_message=str(e),
            )
            raise

        finally:
            self.monitor.close()

    #############################################################


#############################################################


if __name__ == "__main__":

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
        ar.make_paretoish_movie()
        if vae.cfg.make_mu_log_var_movies:
            ar.movie_builder.make_mu_log_var_movie(log_var_graph=True)
            ar.movie_builder.make_mu_log_var_movie(log_var_graph=False)