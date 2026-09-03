"""
Held-out evaluation for a VAE run.

The training loop records per-batch losses on the training set, which makes
every frontier built from them a training-set frontier. This module runs the
model over the whole validation split at a fixed epoch cadence so the same
frontier machinery can be pointed at held-out data.

Two reconstruction losses are reported per evaluation:

    recon_mu        decoder(mu), with the posterior spread removed. This is
                    deterministic and is the expected reconstruction loss.
    recon_sampled   decoder(mu + sigma * eps), matching what the training
                    loop measures. Comparable to the training curve, so the
                    train/val gap computed from it is meaningful.

The gap between the two is itself informative: it grows with the posterior
spread, which is what beta controls.

Randomness is deliberately kept off the global TensorFlow stream. The
encoder's Sampling layer calls tf.keras.backend.random_normal, so running the
full encoder here would consume draws that training would otherwise receive
and shift every subsequent epoch. Instead a submodel is built that stops at
mu and log_var, and the sampled z is formed with tf.random.stateless_normal.
Validation therefore does not perturb training reproducibility.

Also accumulated is the per-dimension aggregate posterior variance,

    Var(mu_d) + E[sigma^2_d]

which equals 1.0 for every dimension when the aggregate posterior matches a
standard normal prior. Generation samples z from the prior, so this quantity
bears on sample quality in a way neither loss term captures.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class ValidationResult:
    """Metrics from one pass over the validation split."""

    n_images: int
    recon_mu: float
    recon_sampled: float
    kl_loss: float
    kl_sum: float
    ssim_mu: float
    ssim_sampled: float
    active_latent_dims: int
    kl_per_dim: np.ndarray
    agg_post_var_per_dim: np.ndarray

    @property
    def agg_post_var_mean(self):
        return float(np.mean(self.agg_post_var_per_dim))

    @property
    def agg_post_var_min(self):
        return float(np.min(self.agg_post_var_per_dim))

    @property
    def agg_post_var_max(self):
        return float(np.max(self.agg_post_var_per_dim))


def make_mu_log_var_net(encoder_net):
    """
    Submodel returning (mu, log_var) without the Sampling layer.

    Cached on the encoder so the graph is built once per run.
    """
    cached = getattr(encoder_net, "_mu_log_var_net", None)
    if cached is not None:
        return cached

    submodel = tf.keras.Model(
        encoder_net.inputs,
        [
            encoder_net.get_layer("latent_mu").output,
            encoder_net.get_layer("latent_log_var").output,
        ],
        name="Encoder_MuLogVar",
    )
    encoder_net._mu_log_var_net = submodel
    return submodel


@tf.function(reduce_retracing=True)
def _validation_batch(
    x_batch, mu_log_var_net, decoder_net, loss_fn, num_input_pixels, seed
):
    """Forward pass only. Returns per-batch sums so partial batches weight correctly."""
    mu, log_var = mu_log_var_net(x_batch, training=False)

    # Stateless draw: reproducible, and independent of the global RNG that
    # the training loop's Sampling layer consumes.
    epsilon = tf.random.stateless_normal(tf.shape(mu), seed=seed, dtype=mu.dtype)
    z = mu + tf.exp(0.5 * log_var) * epsilon

    reconstructed = decoder_net(z, training=False)
    reconstructed_mu = decoder_net(mu, training=False)

    n = tf.cast(tf.shape(x_batch)[0], tf.float32)

    # num_input_pixels arrives as a numpy int64 (np.prod of the encoder input
    # shape). train_step reads it off vae_obj inside the trace, so it is baked
    # in as a Python constant there; passed as an argument it becomes an int64
    # tensor, and float32 * int64 is a TypeError.
    num_input_pixels = tf.cast(num_input_pixels, tf.float32)

    recon_sampled = loss_fn(x_batch, reconstructed) * num_input_pixels
    recon_mu = loss_fn(x_batch, reconstructed_mu) * num_input_pixels

    ssim_sampled = tf.reduce_mean(
        tf.image.ssim(x_batch, reconstructed, max_val=1.0)
    )
    ssim_mu = tf.reduce_mean(
        tf.image.ssim(x_batch, reconstructed_mu, max_val=1.0)
    )

    kl_by_sample_and_dim = -0.5 * (
        1.0 + log_var - tf.square(mu) - tf.exp(log_var)
    )

    return {
        "n": n,
        "recon_mu_sum": recon_mu * n,
        "recon_sampled_sum": recon_sampled * n,
        "ssim_mu_sum": ssim_mu * n,
        "ssim_sampled_sum": ssim_sampled * n,
        "kl_per_dim_sum": tf.reduce_sum(kl_by_sample_and_dim, axis=0),
        "mu_sum": tf.reduce_sum(mu, axis=0),
        "mu_sq_sum": tf.reduce_sum(tf.square(mu), axis=0),
        "sigma_sq_sum": tf.reduce_sum(tf.exp(log_var), axis=0),
    }


def evaluate_validation_set(
    validation_dataset,
    encoder_net,
    decoder_net,
    loss_fn,
    num_input_pixels,
    active_dim_kl_threshold,
    seed=0,
):
    """
    Run the model over the entire validation split and return aggregate metrics.

    Accumulation is by sample rather than by batch, so the final partial batch
    (validation uses drop_remainder=False) is weighted correctly.
    """
    mu_log_var_net = make_mu_log_var_net(encoder_net)

    totals = None
    n_images = 0.0

    for batch_index, x_batch in enumerate(validation_dataset):
        # Same seed pair every evaluation, so epsilon is identical across
        # epochs and recon_sampled changes only because the model changed.
        batch = _validation_batch(
            x_batch,
            mu_log_var_net,
            decoder_net,
            loss_fn,
            num_input_pixels,
            tf.constant([seed, batch_index], dtype=tf.int32),
        )
        batch = {key: value.numpy() for key, value in batch.items()}

        n_images += batch.pop("n")
        if totals is None:
            totals = batch
        else:
            for key, value in batch.items():
                totals[key] = totals[key] + value

    if totals is None or n_images == 0:
        raise ValueError("Validation dataset yielded no batches")

    kl_per_dim = totals["kl_per_dim_sum"] / n_images

    # Aggregate posterior variance per latent dimension:
    #   Var(mu_d) + E[sigma^2_d]
    # equal to 1.0 for every dimension under a perfectly matched prior.
    mu_mean = totals["mu_sum"] / n_images
    mu_var = totals["mu_sq_sum"] / n_images - np.square(mu_mean)
    agg_post_var_per_dim = mu_var + totals["sigma_sq_sum"] / n_images

    return ValidationResult(
        n_images=int(n_images),
        recon_mu=float(totals["recon_mu_sum"] / n_images),
        recon_sampled=float(totals["recon_sampled_sum"] / n_images),
        kl_loss=float(np.mean(kl_per_dim)),
        kl_sum=float(np.sum(kl_per_dim)),
        ssim_mu=float(totals["ssim_mu_sum"] / n_images),
        ssim_sampled=float(totals["ssim_sampled_sum"] / n_images),
        active_latent_dims=int(np.sum(kl_per_dim > active_dim_kl_threshold)),
        kl_per_dim=kl_per_dim,
        agg_post_var_per_dim=agg_post_var_per_dim,
    )
