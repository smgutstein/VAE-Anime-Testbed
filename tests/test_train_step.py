import numpy as np
import pytest


class _CfgStub:
    max_grad_norm = 5.0
    step_guard_kl_jump_ratio_threshold = 100.0
    step_guard_kl_abs_threshold = 1e6
    step_guard_max_log_var_threshold = 20.0
    active_dim_kl_threshold = 1e-2


class _TrainerLikeStub:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.cfg = _CfgStub()


class TestTrainStepIntegration:
    def _tiny_vae(self, tmp_path):
        from VAE_Anime_Full_Model import VAE_Model

        return VAE_Model(
            enc_input_shape=(16, 16, 3),
            latent_dim=4,
            base_filters=8,
            filter_factors=(1, 2, 4),
            encode_dense_units=32,
            kernel_size=3,
            output_dir=str(tmp_path),
        )

    def _make_step_context(self, tmp_path, tf, kl_weight_value=0.5):
        vae_model = self._tiny_vae(tmp_path)
        trainer_like = _TrainerLikeStub(vae_model)

        x = tf.random.uniform((4, 16, 16, 3), dtype=tf.float32)
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        kl_weight = tf.Variable(kl_weight_value, dtype=tf.float32, trainable=False)
        prev_kl = tf.Variable(1.0, dtype=tf.float32, trainable=False)

        # Force model weights to exist before train_step's tf.function runs.
        _ = vae_model.vae_net(x)

        # Force optimizer slot variables to be created outside tf.function.
        optimizer.build(vae_model.vae_net.trainable_weights)

        return trainer_like, x, loss_fn, optimizer, kl_weight, prev_kl

    def _run_step(self, tmp_path, tf, kl_weight_value=0.5):
        from VAE_Anime_TrainStep import train_step

        trainer_like, x, loss_fn, optimizer, kl_weight, prev_kl = \
            self._make_step_context(tmp_path, tf, kl_weight_value=kl_weight_value)

        return train_step(
            x,
            kl_weight,
            prev_kl,
            trainer_like,
            loss_fn,
            optimizer,
        )

    def test_losses_are_finite(self, tmp_path, tf):
        loss_recon, loss_kl, *_ = self._run_step(tmp_path, tf)

        assert np.isfinite(loss_recon.numpy())
        assert np.isfinite(loss_kl.numpy())

    def test_mu_and_log_var_shapes(self, tmp_path, tf):
        _, _, mu, log_var, *_ = self._run_step(tmp_path, tf)

        assert mu.shape == (4, 4)
        assert log_var.shape == (4, 4)

    def test_mu_and_log_var_are_finite(self, tmp_path, tf):
        _, _, mu, log_var, *_ = self._run_step(tmp_path, tf)

        assert bool(tf.reduce_all(tf.math.is_finite(mu)).numpy())
        assert bool(tf.reduce_all(tf.math.is_finite(log_var)).numpy())

    def test_recon_ssim_kl_per_dim_and_active_dims_are_valid(self, tmp_path, tf):
        _, loss_kl, mu, _, recon_ssim, kl_per_dim, active_latent_dims, *_ = \
            self._run_step(tmp_path, tf)

        assert np.isfinite(recon_ssim.numpy())
        assert -1.0 <= float(recon_ssim.numpy()) <= 1.0
        assert kl_per_dim.shape == (mu.shape[1],)
        assert bool(tf.reduce_all(tf.math.is_finite(kl_per_dim)).numpy())
        assert float(tf.reduce_mean(kl_per_dim).numpy()) == pytest.approx(
            float(loss_kl.numpy())
        )
        assert 0 <= int(active_latent_dims.numpy()) <= mu.shape[1]

    def test_kl_weight_zero_does_not_cause_nan(self, tmp_path, tf):
        loss_recon, loss_kl, *_ = self._run_step(tmp_path, tf, kl_weight_value=0.0)

        assert np.isfinite(loss_recon.numpy())
        assert np.isfinite(loss_kl.numpy())

    def test_nominal_step_is_not_toxic_and_applies_update(self, tmp_path, tf):
        *_, toxic_step, toxic_reasons, applied_update = self._run_step(tmp_path, tf)

        assert bool(toxic_step.numpy()) is False
        assert toxic_reasons.shape[0] == 0
        assert bool(applied_update.numpy()) is True

    def test_build_step_result_converts_raw_outputs(self, tmp_path, tf):
        from VAE_Anime_TrainStep import build_step_result

        raw = self._run_step(tmp_path, tf)
        x_batch_train = tf.zeros((4, 16, 16, 3), dtype=tf.float32)

        result = build_step_result(
            raw_step_output=raw,
            x_batch_train=x_batch_train,
            epoch=3,
            step=7,
            kl_weight=0.5,
        )

        assert result.epoch == 3
        assert result.step == 7
        assert result.kl_weight == pytest.approx(0.5)
        assert isinstance(result.loss_recon, float)
        assert isinstance(result.loss_kl, float)
        assert isinstance(result.recon_ssim, float)
        assert isinstance(result.kl_per_dim, tuple)
        assert isinstance(result.active_latent_dims, int)
        assert len(result.kl_per_dim) == result.mu.shape[1]
        assert 0 <= result.active_latent_dims <= result.mu.shape[1]
        assert isinstance(result.toxic_step, bool)
        assert isinstance(result.toxic_reasons, tuple)
        assert isinstance(result.applied_update, bool)
        assert result.mu.shape == (4, 4)
        assert result.log_var.shape == (4, 4)