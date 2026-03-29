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
        Return the current KL adjustment factor / weight.
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
                "adj_str": ...,
                "factor": ...,
                "num_maxes": ...,
                "test1": ...,
                "test2": ...,
                "max_factor_seen": ...,
                "min_factor_seen": ...,
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
            "kl_adj_factor": self.current_value(),
            "kl_adj_update_factor": self.current_update_factor(),
        }


class AdaptiveKLLossPolicy(BaseLossPolicy):
    """
    Thin wrapper around the existing KLController so the trainer depends on
    a policy interface instead of on KLController directly.
    """

    def __init__(self, kl_adj_factor, kl_adj_factor_max,
                 kl_adj_update_factor, running_window):
        self.kl_controller = KLController(
            initial_factor=kl_adj_factor,
            max_factor=kl_adj_factor_max,
            update_factor=kl_adj_update_factor,
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

    Internally still uses the legacy variable names so the rest of the code
    can keep calling the result a kl_adj_factor for now.
    """

    def __init__(self, kl_adj_factor):
        self.kl_adj_factor = float(kl_adj_factor)
        self.kl_adj_factor_max = float(kl_adj_factor)
        self.kl_adj_update_factor = 0.0
        self.running_window = 1

    @property
    def policy_name(self):
        return "fixed_beta"

    def current_value(self):
        return self.kl_adj_factor

    def current_update_factor(self):
        return self.kl_adj_update_factor

    def update(self, curr_loss_recon, curr_loss_kl):
        """
        No adaptation. Return an update_info dict shaped like the adaptive
        controller's dict so existing trainer/monitor code can keep working.
        """
        return {
            "policy_name": self.policy_name,
            "adj_str": "=",
            "factor": self.kl_adj_factor,
            "num_maxes": 0,
            "test1": False,
            "test2": False,
            "max_factor_seen": self.kl_adj_factor,
            "min_factor_seen": self.kl_adj_factor,
            "window_len": 1,
            "update_factor": self.kl_adj_update_factor,
            "update_factor_changed": False,
        }


def build_loss_policy(cfg):
    """
    Factory to construct the loss policy from config.

    Expected config additions:
        cfg.loss_policy in {"adaptive_kl", "fixed_beta"}
        cfg.beta  (or, if you prefer less churn, cfg.fixed_kl_adj_factor)

    To keep variable names consistent with the current codebase, the
    fixed-beta path still feeds the value in as a kl_adj_factor.
    """
    if cfg.loss_policy == "adaptive_kl":
        return AdaptiveKLLossPolicy(
            kl_adj_factor=cfg.kl_adj_factor,
            kl_adj_factor_max=cfg.kl_adj_factor_max,
            kl_adj_update_factor=cfg.kl_adj_update_factor,
            running_window=cfg.running_window,
        )

    #if cfg.loss_policy == "fixed_beta":
    #    return FixedBetaLossPolicy(
    #        kl_adj_factor=cfg.beta,
    #    )

    raise ValueError(f"Unknown loss_policy: {cfg.loss_policy}")