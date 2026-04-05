import pytest


class DummyLossPolicy:
    policy_name = "adaptive_kl"

    def __init__(self, value=0.5):
        self._value = float(value)

    def current_value(self):
        return self._value


def _make_step_result(tf, **overrides):
    from VAE_Anime_StepGuard import StepResult

    base = dict(
        epoch=2,
        step=5,
        kl_weight=0.5,
        x_batch_train=tf.zeros((2, 8, 8, 3), dtype=tf.float32),
        mu=tf.zeros((2, 4), dtype=tf.float32),
        log_var=tf.zeros((2, 4), dtype=tf.float32),
        loss_recon=10.0,
        loss_kl=0.25,
        grad_norm=1.0,
        max_log_var=0.0,
        min_log_var=0.0,
        max_abs_mu=0.0,
        kl_jump_ratio=1.0,
        toxic_step=False,
        applied_update=True,
        mu_finite=True,
        log_var_finite=True,
    )
    base.update(overrides)
    return StepResult(**base)


def test_clean_applied_step_updates_prev_kl_and_records_recent_step(tmp_path, tf):
    from VAE_Anime_StepGuard import StepGuard

    guard = StepGuard(
        stats_dir=tmp_path,
        loss_policy=DummyLossPolicy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    )

    result = _make_step_result(tf, loss_kl=0.75, applied_update=True, toxic_step=False)
    decision = guard.handle_step(result)

    assert decision.should_continue is False
    assert decision.should_raise is False
    assert decision.should_update_prev_kl is True
    assert decision.prev_kl_value == pytest.approx(0.75)
    assert len(guard.recent_good_steps) == 1


def test_tripwire_skip_halves_lr_requests_continue_and_saves_artifact(tmp_path, monkeypatch, tf):
    import VAE_Anime_StepGuard as sg
    from VAE_Anime_StepGuard import StepGuard

    save_calls = []
    monkeypatch.setattr(sg, "save_failure_tensors", lambda **kwargs: save_calls.append(kwargs))

    guard = StepGuard(
        stats_dir=tmp_path,
        loss_policy=DummyLossPolicy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    )

    result = _make_step_result(tf, toxic_step=True, applied_update=False, kl_jump_ratio=250.0, max_log_var=21.0)
    decision = guard.handle_step(result)

    assert decision.should_continue is True
    assert decision.should_raise is False
    assert decision.should_update_prev_kl is False
    assert float(guard.optimizer.learning_rate.numpy()) == pytest.approx(5e-4)
    assert len(save_calls) == 1
    assert save_calls[0]["file_tag"] == "tripwire_skip"


def test_divergence_with_recent_good_step_saves_last_good_and_history(tmp_path, monkeypatch, tf):
    import VAE_Anime_StepGuard as sg
    from VAE_Anime_StepGuard import StepGuard, TrainingDivergedError

    tensor_calls = []
    history_calls = []

    monkeypatch.setattr(sg, "save_failure_tensors", lambda **kwargs: tensor_calls.append(kwargs))
    monkeypatch.setattr(sg, "save_failure_history", lambda **kwargs: history_calls.append(kwargs))

    guard = StepGuard(
        stats_dir=tmp_path,
        loss_policy=DummyLossPolicy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    )

    good = _make_step_result(tf, epoch=1, step=2, loss_recon=9.0, loss_kl=0.2)
    good_decision = guard.handle_step(good)
    assert good_decision.should_raise is False
    assert len(guard.recent_good_steps) == 1

    bad = _make_step_result(tf, epoch=1, step=3, loss_recon=float("nan"), mu_finite=False, applied_update=False)
    bad_decision = guard.handle_step(bad)

    assert bad_decision.should_raise is True
    assert isinstance(bad_decision.exception, TrainingDivergedError)
    assert len(tensor_calls) == 1
    assert tensor_calls[0]["file_tag"] == "last_good_before_crash"
    assert len(history_calls) == 1
