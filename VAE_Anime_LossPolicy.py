from abc import ABC, abstractmethod

from VAE_Anime_KL_Controller import KLController


class BaseLossPolicy(ABC):
    """
    Abstract interface for KL-weight / loss-weight policies.

    Kept intentionally close to the existing KLController API so that
    VAE_Anime_Train.py can be refactored with minimal variable churn.
    """

    @property
    @abstractmethod
    def policy_name(self):
        pass

    @abstractmethod
    def current_value(self):
        """
        Return the current KL weight.
        """
        pass

    @abstractmethod
    def current_update_factor(self):
        """
        Return the current multiplicative update factor used by the policy.

        For non-adaptive policies this can simply return 0.0 or 1.0,
        depending on what is most convenient for logging.
        """
        pass

    @abstractmethod
    def update(self, curr_loss_recon, curr_loss_kl):
        """
        Update internal policy state after one train step.

        Must return a dict compatible with the existing trainer usage:
            {
                "weight_direction": ...,
                "kl_weight": ...,
                "num_maxes": ...,
                "test1": ...,
                "test2": ...,
                "max_kl_weight_seen": ...,
                "min_kl_weight_seen": ...,
                "window_len": ...,
                "update_factor": ...,
                "update_factor_changed": ...,
            }
        """
        pass

    def text_fields(self):
        """
        Optional helper for future logging / run summaries.
        """
        return {
            "policy_name": self.policy_name,
            "kl_weight": self.current_value(),
            "kl_weight_update_factor": self.current_update_factor(),
        }


class AdaptiveKLLossPolicy(BaseLossPolicy):
    """
    Thin wrapper around the existing KLController so the trainer depends on
    a policy interface instead of on KLController directly.
    """

    def __init__(self, initial_kl_weight, max_kl_weight,
                 kl_weight_update_factor, running_window):
        self.kl_controller = KLController(
            initial_kl_weight=initial_kl_weight,
            max_kl_weight=max_kl_weight,
            kl_weight_update_factor=kl_weight_update_factor,
            running_window=running_window,
        )

    @property
    def policy_name(self):
        return "adaptive_kl"

    def current_value(self):
        return self.kl_controller.current_value()

    def current_update_factor(self):
        return self.kl_controller.current_update_factor()

    def update(self, curr_loss_recon, curr_loss_kl):
        update_info = self.kl_controller.update(curr_loss_recon, curr_loss_kl)
        update_info["policy_name"] = self.policy_name
        return update_info


class FixedBetaLossPolicy(BaseLossPolicy):
    """
    Fixed-beta / fixed-KL-weight baseline.
    """

    def __init__(self, beta):
        self.beta = float(beta)
        self.beta_max = float(beta)
        self.kl_weight_update_factor = 0.0
        self.running_window = 1

    @property
    def policy_name(self):
        return "fixed_beta"

    def current_value(self):
        return self.beta

    def current_update_factor(self):
        return self.kl_weight_update_factor

    def update(self, curr_loss_recon, curr_loss_kl):
        """
        No adaptation. Return an update_info dict shaped like the adaptive
        controller's dict so existing trainer/monitor code can keep working.
        """
        return {
            "policy_name": self.policy_name,
            "weight_direction": "=",
            "kl_weight": self.beta,
            "num_maxes": 0,
            "test1": False,
            "test2": False,
            "max_kl_weight_seen": self.beta,
            "min_kl_weight_seen": self.beta,
            "window_len": 1,
            "update_factor": self.kl_weight_update_factor,
            "update_factor_changed": False,
        }


def build_loss_policy(cfg):
    """
    Factory to construct the loss policy from config.

    Expected config additions:
        cfg.loss_policy in {"adaptive_kl", "fixed_beta"}

    Adaptive mode reads the KL-weight schedule fields from config;
    fixed-beta mode reads cfg.beta.
    """
    if cfg.loss_policy == "adaptive_kl":
        return AdaptiveKLLossPolicy(
            initial_kl_weight=cfg.initial_kl_weight,
            max_kl_weight=cfg.max_kl_weight,
            kl_weight_update_factor=cfg.kl_weight_update_factor,
            running_window=cfg.running_window,
        )

    if cfg.loss_policy == "fixed_beta":
        return FixedBetaLossPolicy(
            beta=cfg.beta,
        )

    raise ValueError(f"Unknown loss_policy: {cfg.loss_policy}")