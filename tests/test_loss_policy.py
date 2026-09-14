import math
from types import SimpleNamespace

import pytest


class TestAdaptiveKLWeightScheduler:
    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_Anime_KL_Weight_Scheduler import AdaptiveKLWeightScheduler
        self.Scheduler = AdaptiveKLWeightScheduler

    def _make(self, initial=0.01, max_kl=1.0, factor=0.1, window=10):
        return self.Scheduler(initial_kl_weight=initial, max_kl_weight=max_kl, kl_weight_update_factor=factor, running_window=window)

    def test_weight_increases_when_recon_improves(self):
        s = self._make(initial=0.5, max_kl=100.0)
        w0 = s.current_value()
        assert s.update(curr_loss_recon=0.8, curr_loss_kl=0.1)["kl_weight"] > w0

    def test_weight_decreases_when_recon_gets_worse(self):
        s = self._make(initial=0.5, max_kl=100.0)
        s.update(curr_loss_recon=0.5, curr_loss_kl=0.1)
        w1 = s.current_value()
        assert s.update(curr_loss_recon=0.9, curr_loss_kl=0.1)["kl_weight"] < w1

    def test_weight_never_exceeds_max(self):
        s = self._make(initial=0.9, max_kl=1.0, factor=0.5)
        for i in range(20):
            s.update(curr_loss_recon=0.01 * i, curr_loss_kl=0.0)
        assert s.current_value() <= 1.0

    def test_update_factor_is_90pct_of_prior_after_trigger(self):
        s = self._make(initial=0.99, max_kl=1.0, factor=0.1, window=10)
        factor_before = s.current_update_factor()
        info = s.update(curr_loss_recon=0.1, curr_loss_kl=0.1)
        assert info["update_factor_changed"]
        assert math.isclose(s.current_update_factor(), factor_before * 0.9, rel_tol=1e-6)


class TestLossPolicies:
    def test_fixed_beta_policy(self):
        from VAE_Anime_LossPolicy import FixedBetaLossPolicy

        p = FixedBetaLossPolicy(0.42)
        assert p.current_value() == pytest.approx(0.42)
        assert p.current_update_factor() == pytest.approx(0.0)
        info = p.update(0.5, 0.1)
        assert info["policy_name"] == "fixed_beta"
        assert info["weight_direction"] == "="
        assert info["kl_weight"] == pytest.approx(0.42)

    def test_adaptive_kl_policy(self):
        from VAE_Anime_LossPolicy import AdaptiveKLLossPolicy

        p = AdaptiveKLLossPolicy(0.01, 1.0, 0.1, 10)
        info = p.update(0.5, 0.1)
        assert info["policy_name"] == "adaptive_kl"
        assert p.current_value() > 0.01

    def test_adaptive_policy_state_round_trip(self):
        from VAE_Anime_LossPolicy import AdaptiveKLLossPolicy

        original = AdaptiveKLLossPolicy(0.01, 1.0, 0.1, 10)
        original.update(0.5, 0.1)
        original.update(0.6, 0.2)
        restored = AdaptiveKLLossPolicy(0.01, 1.0, 0.1, 10)
        restored.set_state(original.get_state())

        assert restored.get_state() == original.get_state()

    def test_build_loss_policy(self):
        from VAE_Anime_LossPolicy import build_loss_policy, AdaptiveKLLossPolicy, FixedBetaLossPolicy

        adaptive_cfg = SimpleNamespace(
            loss_policy="adaptive_kl",
            beta=None,
            initial_kl_weight=0.01,
            max_kl_weight=1.0,
            kl_weight_update_factor=0.1,
            running_window=10,
        )
        fixed_cfg = SimpleNamespace(
            loss_policy="fixed_beta",
            beta=0.5,
            initial_kl_weight=None,
            max_kl_weight=None,
            kl_weight_update_factor=None,
            running_window=None,
        )

        assert isinstance(build_loss_policy(adaptive_cfg), AdaptiveKLLossPolicy)
        assert isinstance(build_loss_policy(fixed_cfg), FixedBetaLossPolicy)

        with pytest.raises(ValueError, match="Unknown loss_policy"):
            build_loss_policy(SimpleNamespace(loss_policy="magic", beta=None, initial_kl_weight=None, max_kl_weight=None, kl_weight_update_factor=None, running_window=None))
