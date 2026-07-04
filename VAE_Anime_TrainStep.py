import tensorflow as tf

from VAE_Anime_StepGuard import StepResult


@tf.function(reduce_retracing=True)
def train_step(
    x_batch_train,
    beta_factor,
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
        recon_ssim = tf.reduce_mean(
            tf.image.ssim(x_batch_train, reconstructed, max_val=1.0)
        )

        kl_by_sample_and_dim = -0.5 * (
            1.0 + log_var - tf.square(mu) - tf.exp(log_var)
        )
        kl_per_dim = tf.reduce_mean(kl_by_sample_and_dim, axis=0)
        active_dim_kl_threshold = tf.constant(
            vae_obj.cfg.active_dim_kl_threshold,
            dtype=kl_per_dim.dtype,
        )
        active_latent_dims = tf.reduce_sum(
            tf.cast(kl_per_dim > active_dim_kl_threshold, tf.int32)
        )
        loss_kl = tf.reduce_mean(kl_per_dim)
        loss_tot = loss_recon + beta_factor * loss_kl

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
        recon_ssim,
        kl_per_dim,
        active_latent_dims,
        grad_norm,
        max_log_var,
        min_log_var,
        max_abs_mu,
        kl_jump_ratio,
        toxic_step,
        toxic_reasons,
        applied_update,
    )


def build_step_result(raw_step_output, x_batch_train, epoch, step, kl_weight):
    (
        loss_recon,
        loss_kl,
        mu,
        log_var,
        recon_ssim,
        kl_per_dim,
        active_latent_dims,
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
        recon_ssim=float(recon_ssim.numpy()),
        kl_per_dim=tuple(float(x) for x in kl_per_dim.numpy().tolist()),
        active_latent_dims=int(active_latent_dims.numpy()),
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