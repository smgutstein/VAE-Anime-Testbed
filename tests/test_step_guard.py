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
        recon_ssim=0.5,
        kl_per_dim=(0.1, 0.2, 0.3, 0.4),
        active_latent_dims=4,
        grad_norm=1.0,
        max_log_var=0.0,
        min_log_var=0.0,
        max_abs_mu=0.0,
        kl_jump_ratio=1.0,
        toxic_step=False,
        toxic_reasons=(),
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

    result = _make_step_result(
        tf,
        toxic_step=True,
        toxic_reasons=("kl_jump_ratio", "max_log_var"),
        applied_update=False,
        kl_jump_ratio=250.0,
        max_log_var=21.0,
    )
    decision = guard.handle_step(result)

    assert decision.should_continue is True
    assert decision.should_raise is False
    assert decision.should_update_prev_kl is False
    assert float(guard.optimizer.learning_rate.numpy()) == pytest.approx(5e-4)
    assert len(save_calls) == 1
    assert save_calls[0]["file_tag"] == "tripwire_skip"

    # Important regression check:
    assert len(guard.recent_good_steps) == 0

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

def test_repeated_tripwires_raise_after_threshold(tmp_path, monkeypatch, tf):
    import VAE_Anime_StepGuard as sg
    from VAE_Anime_StepGuard import StepGuard, TrainingDivergedError

    save_calls = []

    monkeypatch.setattr(
        sg,
        "save_failure_tensors",
        lambda **kwargs: save_calls.append(kwargs),
    )

    guard = StepGuard(
        stats_dir=tmp_path,
        loss_policy=DummyLossPolicy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        max_consecutive_tripwires=3,
    )

    result = _make_step_result(
        tf,
        toxic_step=True,
        toxic_reasons=("kl_jump_ratio", "max_log_var"),
        applied_update=False,
        kl_jump_ratio=250.0,
        max_log_var=21.0,
    )

    d1 = guard.handle_step(result)
    assert d1.should_continue is True
    assert d1.should_raise is False

    d2 = guard.handle_step(result)
    assert d2.should_continue is True
    assert d2.should_raise is False

    d3 = guard.handle_step(result)
    assert d3.should_raise is True
    assert isinstance(d3.exception, TrainingDivergedError)

    assert len(save_calls) == 3
    assert float(guard.optimizer.learning_rate.numpy()) == pytest.approx(1.25e-4)

def test_toxic_skipped_step_is_not_recorded_as_recent_good(tmp_path, monkeypatch, tf):
    import VAE_Anime_StepGuard as sg
    from VAE_Anime_StepGuard import StepGuard

    monkeypatch.setattr(sg, "save_failure_tensors", lambda **kwargs: None)

    guard = StepGuard(
        stats_dir=tmp_path,
        loss_policy=DummyLossPolicy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    )

    result = _make_step_result(
        tf,
        toxic_step=True,
        toxic_reasons=("kl_jump_ratio",),
        applied_update=False,
        kl_jump_ratio=250.0,
    )

    decision = guard.handle_step(result)

    assert decision.should_continue is True
    assert len(guard.recent_good_steps) == 0

class _FakeLR:
    def __init__(self, value):
        self.value = float(value)

    def numpy(self):
        return self.value

    def assign(self, new_value):
        self.value = float(new_value)


class _FakeOptimizer:
    def __init__(self, lr):
        self.learning_rate = _FakeLR(lr)


class TestTripwireReport:
    """
    The learning-rate backoff has no restore path, so run_summary.json's
    "learning_rate" (the configured value) can be orders of magnitude above
    what most of a run actually used. These cumulative counters make that
    visible instead of leaving it implicit.
    """

    @staticmethod
    def _guard(lr=2e-3, backoff=0.5, floor=1e-6):
        from VAE_Anime_StepGuard import StepGuard
        guard = StepGuard.__new__(StepGuard)
        guard.optimizer = _FakeOptimizer(lr)
        guard.lr_backoff = backoff
        guard.lr_floor = floor
        guard.consecutive_tripwires = 0
        guard.total_tripwires = 0
        guard.tripwire_events = []
        guard.initial_learning_rate = lr
        guard.min_learning_rate_seen = lr
        return guard

    @staticmethod
    def _trip(guard, epoch=0, step=0):
        """The learning-rate and bookkeeping half of _handle_tripwire_skip."""
        guard.consecutive_tripwires += 1
        guard.total_tripwires += 1
        old_lr = float(guard.optimizer.learning_rate.numpy())
        new_lr = max(old_lr * guard.lr_backoff, guard.lr_floor)
        guard.optimizer.learning_rate.assign(new_lr)
        guard.min_learning_rate_seen = min(guard.min_learning_rate_seen, new_lr)
        guard.tripwire_events.append({
            "epoch": int(epoch),
            "step": int(step),
            "consecutive": int(guard.consecutive_tripwires),
            "learning_rate_before": old_lr,
            "learning_rate_after": new_lr,
            "kl_jump_ratio": 12.5,
            "grad_norm": 900.0,
            "max_log_var": 4.0,
            "kl_weight": 417.3,
        })

    def test_clean_run_reports_no_backoff(self):
        report = self._guard().tripwire_report()
        assert report["total_tripwires"] == 0
        assert report["tripwire_events"] == []
        assert report["first_tripwire_epoch"] is None
        assert report["last_tripwire_epoch"] is None
        assert report["learning_rate_backoff_applied"] is False
        assert report["final_learning_rate"] == report["initial_learning_rate"]

    def test_total_survives_the_consecutive_counter_resetting(self):
        guard = self._guard()
        for _ in range(3):
            self._trip(guard)
            guard.consecutive_tripwires = 0   # an accepted step

        report = guard.tripwire_report()
        assert guard.consecutive_tripwires == 0
        assert report["total_tripwires"] == 3

    def test_backoff_is_permanent_across_isolated_trips(self):
        guard = self._guard(lr=2e-3)
        for _ in range(4):
            self._trip(guard)
            guard.consecutive_tripwires = 0

        report = guard.tripwire_report()
        assert report["final_learning_rate"] == pytest.approx(2e-3 / 16)
        assert report["learning_rate_backoff_applied"] is True

    def test_learning_rate_stops_at_the_floor(self):
        guard = self._guard(lr=2e-3, floor=1e-6)
        for _ in range(50):
            self._trip(guard)

        report = guard.tripwire_report()
        assert report["final_learning_rate"] == pytest.approx(1e-6)
        assert report["min_learning_rate_seen"] == pytest.approx(1e-6)
        assert report["total_tripwires"] == 50


    def test_each_firing_records_its_epoch_and_step(self):
        guard = self._guard()
        self._trip(guard, epoch=106, step=30)
        guard.consecutive_tripwires = 0
        self._trip(guard, epoch=107, step=5)

        events = guard.tripwire_report()["tripwire_events"]
        assert [(e["epoch"], e["step"]) for e in events] == [(106, 30), (107, 5)]
        assert [e["consecutive"] for e in events] == [1, 1]

    def test_first_and_last_epoch_bracket_the_events(self):
        guard = self._guard()
        for epoch in (12, 480, 4931):
            self._trip(guard, epoch=epoch, step=7)
            guard.consecutive_tripwires = 0

        report = guard.tripwire_report()
        assert report["first_tripwire_epoch"] == 12
        assert report["last_tripwire_epoch"] == 4931
        assert report["total_tripwires"] == 3

    def test_events_carry_the_learning_rate_on_both_sides(self):
        guard = self._guard(lr=2e-3)
        self._trip(guard, epoch=50, step=1)
        self._trip(guard, epoch=50, step=2)

        events = guard.tripwire_report()["tripwire_events"]
        assert events[0]["learning_rate_before"] == pytest.approx(2e-3)
        assert events[0]["learning_rate_after"] == pytest.approx(1e-3)
        assert events[1]["learning_rate_before"] == pytest.approx(1e-3)
        assert events[1]["learning_rate_after"] == pytest.approx(5e-4)
        # Consecutive trips, so the counter climbs.
        assert [e["consecutive"] for e in events] == [1, 2]

    def test_report_returns_a_copy_of_the_event_list(self):
        guard = self._guard()
        self._trip(guard, epoch=1, step=1)
        report = guard.tripwire_report()
        report["tripwire_events"].append({"epoch": 999})
        assert len(guard.tripwire_events) == 1
