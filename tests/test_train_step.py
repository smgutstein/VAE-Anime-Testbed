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

    def test_sampling_is_identical_at_same_training_position(self, tmp_path, tf):
        vae = self._tiny_vae(tmp_path)
        x = tf.ones((4, 16, 16, 3), dtype=tf.float32)
        vae.encoder.sampling_layer.set_training_position(7, 3)
        first = vae.vae_net(x, training=False)[0]
        vae.encoder.sampling_layer.set_training_position(7, 3)
        second = vae.vae_net(x, training=False)[0]
        np.testing.assert_array_equal(first.numpy(), second.numpy())

    def test_sampling_changes_at_different_training_positions(self, tmp_path, tf):
        vae = self._tiny_vae(tmp_path)
        x = tf.ones((4, 16, 16, 3), dtype=tf.float32)
        vae.encoder.sampling_layer.set_training_position(7, 3)
        first = vae.vae_net(x, training=False)[0]
        vae.encoder.sampling_layer.set_training_position(7, 4)
        second = vae.vae_net(x, training=False)[0]
        assert not np.array_equal(first.numpy(), second.numpy())

    def test_sampling_seed_state_does_not_change_model_weight_layout(self, tmp_path, tf):
        vae = self._tiny_vae(tmp_path)
        assert vae.encoder.sampling_layer.weights == []

    def test_checkpoint_resume_matches_uninterrupted_next_step(self, tmp_path, tf):
        from VAE_Anime_TrainStep import train_step

        x0 = tf.reshape(
            tf.linspace(0.0, 1.0, 4 * 16 * 16 * 3),
            (4, 16, 16, 3),
        )
        x1 = tf.reverse(x0, axis=[0])

        uninterrupted = self._tiny_vae(tmp_path / "uninterrupted")
        uninterrupted_context = _TrainerLikeStub(uninterrupted)
        optimizer_a = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer_a.build(uninterrupted.vae_net.trainable_weights)
        beta_a = tf.Variable(0.5, trainable=False)
        prev_a = tf.Variable(1.0, trainable=False)

        uninterrupted.encoder.sampling_layer.set_training_position(0, 0)
        first = train_step(
            x0, beta_a, prev_a, uninterrupted_context,
            tf.keras.losses.MeanSquaredError(), optimizer_a,
        )
        prev_a.assign(first[1])
        checkpoint = tf.train.Checkpoint(
            vae=uninterrupted.vae_net,
            optimizer=optimizer_a,
            beta_factor=beta_a,
            prev_kl=prev_a,
        )
        checkpoint_path = checkpoint.write(str(tmp_path / "state"))
        checkpointed_weights = [
            variable.numpy().copy() for variable in uninterrupted.vae_net.weights
        ]
        checkpointed_optimizer = [
            variable.numpy().copy() for variable in optimizer_a.variables
        ]

        uninterrupted.encoder.sampling_layer.set_training_position(1, 0)
        train_step(
            x1, beta_a, prev_a, uninterrupted_context,
            tf.keras.losses.MeanSquaredError(), optimizer_a,
        )

        resumed = self._tiny_vae(tmp_path / "resumed")
        resumed_context = _TrainerLikeStub(resumed)
        optimizer_b = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer_b.build(resumed.vae_net.trainable_weights)
        beta_b = tf.Variable(0.5, trainable=False)
        prev_b = tf.Variable(1.0, trainable=False)
        status = tf.train.Checkpoint(
            vae=resumed.vae_net,
            optimizer=optimizer_b,
            beta_factor=beta_b,
            prev_kl=prev_b,
        ).read(checkpoint_path)
        status.assert_existing_objects_matched()

        for expected, actual in zip(checkpointed_weights, resumed.vae_net.weights):
            np.testing.assert_array_equal(expected, actual.numpy())
        for expected, actual in zip(checkpointed_optimizer, optimizer_b.variables):
            np.testing.assert_array_equal(expected, actual.numpy())
        np.testing.assert_array_equal(beta_a.numpy(), beta_b.numpy())
        np.testing.assert_array_equal(prev_a.numpy(), prev_b.numpy())

        resumed.encoder.sampling_layer.set_training_position(1, 0)
        train_step(
            x1, beta_b, prev_b, resumed_context,
            tf.keras.losses.MeanSquaredError(), optimizer_b,
        )

        for expected, actual in zip(
            uninterrupted.vae_net.weights,
            resumed.vae_net.weights,
        ):
            np.testing.assert_allclose(
                expected.numpy(), actual.numpy(), rtol=1e-4, atol=1e-7
            )
        for expected, actual in zip(optimizer_a.variables, optimizer_b.variables):
            np.testing.assert_allclose(
                expected.numpy(), actual.numpy(), rtol=1e-4, atol=1e-7
            )
