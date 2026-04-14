from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from typing import Optional

import numpy as np
import tensorflow as tf

from VAE_Anime_RunArtifacts import (
    latent_diagnostics,
    save_failure_history,
    save_failure_tensors,
)


@dataclass
class StepResult:
    epoch: int
    step: int
    kl_weight: float

    x_batch_train: tf.Tensor
    mu: tf.Tensor
    log_var: tf.Tensor

    loss_recon: float
    loss_kl: float

    grad_norm: float
    max_log_var: float
    min_log_var: float
    max_abs_mu: float
    kl_jump_ratio: float

    toxic_step: bool
    toxic_reasons: tuple[str, ...] = field(default_factory=tuple)
    applied_update: bool = False
    mu_finite: bool = True
    log_var_finite: bool = True


@dataclass
class GuardDecision:
    should_continue: bool = False
    should_raise: bool = False
    exception: Optional[Exception] = None

    should_update_prev_kl: bool = False
    prev_kl_value: Optional[float] = None


class TrainingDivergedError(RuntimeError):
    pass


class StepGuard:
    """
    Python-side bad-step policy:
      - keep recent finite steps
      - save suspicious finite steps
      - save crash artifacts/history
      - apply LR brake when trip-wire fires
      - tell trainer whether to continue, raise, or update prev_kl
    """

    def __init__(self, *, stats_dir, loss_policy, optimizer,
                 recent_good_maxlen: int = 8,
                 lr_backoff: float = 0.5,
                 lr_floor: float = 1e-6,
                 max_consecutive_tripwires: int = 3,
                 suspicious_kl_threshold: float = 1e6,
                 suspicious_abs_mu_threshold: float = 1e4,
                 suspicious_abs_log_var_threshold: float = 50.0,
                 kl_jump_ratio_threshold: float = 100.0,
                 kl_abs_threshold: float = 1e6,
                 max_log_var_threshold: float = 20.0,):
        self.stats_dir = stats_dir
        self.loss_policy = loss_policy
        self.optimizer = optimizer

        self.recent_good_steps = deque(maxlen=recent_good_maxlen)

        self.lr_backoff = float(lr_backoff)
        self.lr_floor = float(lr_floor)
        self.max_consecutive_tripwires = int(max_consecutive_tripwires)
        self.consecutive_tripwires = 0

        self.suspicious_kl_threshold = float(suspicious_kl_threshold)
        self.suspicious_abs_mu_threshold = float(suspicious_abs_mu_threshold)
        self.suspicious_abs_log_var_threshold = float(suspicious_abs_log_var_threshold)
        self.kl_jump_ratio_threshold = float(kl_jump_ratio_threshold)
        self.kl_abs_threshold = float(kl_abs_threshold)
        self.max_log_var_threshold = float(max_log_var_threshold)

    def handle_step(self, result: StepResult) -> GuardDecision:
        loss_bad = self._loss_bad(result)
        latent_bad = self._latent_bad(result)

        if (not loss_bad) and (not latent_bad):
            self._record_good_step(result)

        if self._is_suspicious(result, loss_bad=loss_bad, latent_bad=latent_bad):
            self._save_suspicious_step(result)

        if loss_bad or latent_bad:
            return self._handle_divergence(
                result=result,
                loss_bad=loss_bad,
                latent_bad=latent_bad,
            )

        if result.toxic_step:
            return self._handle_tripwire_skip(result)
        
        self.consecutive_tripwires = 0

        if result.applied_update and np.isfinite(result.loss_kl) and result.loss_kl > 0.0:
            return GuardDecision(
                should_update_prev_kl=True,
                prev_kl_value=float(result.loss_kl),
            )

        return GuardDecision()

    def _loss_bad(self, result: StepResult) -> bool:
        return (
            np.isnan(result.loss_recon)
            or np.isnan(result.loss_kl)
            or np.isinf(result.loss_recon)
            or np.isinf(result.loss_kl)
        )

    def _latent_bad(self, result: StepResult) -> bool:
        return (not result.mu_finite) or (not result.log_var_finite)

    def _is_suspicious(
        self,
        result: StepResult,
        *,
        loss_bad: bool,
        latent_bad: bool,
    ) -> bool:
        if loss_bad or latent_bad:
            return False

        return (
            result.loss_kl > self.suspicious_kl_threshold
            or result.max_abs_mu > self.suspicious_abs_mu_threshold
            or max(abs(result.max_log_var), abs(result.min_log_var))
            > self.suspicious_abs_log_var_threshold
        )

    def _record_good_step(self, result: StepResult) -> None:
        self.recent_good_steps.append(
            {
                "epoch": int(result.epoch),
                "step": int(result.step),
                "x_batch_train": tf.identity(result.x_batch_train),
                "mu": tf.identity(result.mu),
                "log_var": tf.identity(result.log_var),
                "curr_loss_recon": float(result.loss_recon),
                "curr_loss_kl": float(result.loss_kl),
                "kl_weight": float(result.kl_weight),
            }
        )

    def _save_suspicious_step(self, result: StepResult) -> None:
        diagnostics = latent_diagnostics(result.mu, result.log_var)
        diagnostics["note"] = "Suspicious finite step saved before divergence"
        diagnostics["saved_epoch"] = int(result.epoch)
        diagnostics["saved_step"] = int(result.step)
        diagnostics["saved_kl_weight"] = float(result.kl_weight)

        save_failure_tensors(
            stats_dir=self.stats_dir,
            loss_policy=self.loss_policy,
            epoch=result.epoch,
            step=result.step,
            x_batch_train=result.x_batch_train,
            mu=result.mu,
            log_var=result.log_var,
            curr_loss_recon=result.loss_recon,
            curr_loss_kl=result.loss_kl,
            diagnostics=diagnostics,
            file_tag="suspicious",
        )

    def _handle_divergence(
        self,
        *,
        result: StepResult,
        loss_bad: bool,
        latent_bad: bool,
    ) -> GuardDecision:
        diagnostics = latent_diagnostics(result.mu, result.log_var)

        logging.error(
            "Training diverged at epoch=%d step=%d recon=%s kl=%s "
            "kl_weight=%s diagnostics=%s",
            result.epoch,
            result.step,
            result.loss_recon,
            result.loss_kl,
            result.kl_weight,
            diagnostics,
        )

        if self.recent_good_steps:
            last_good = self.recent_good_steps[-1]
            diagnostics["note"] = (
                "Crash detected; saved last known finite step instead of crashing step"
            )
            diagnostics["crash_epoch"] = int(result.epoch)
            diagnostics["crash_step"] = int(result.step)
            diagnostics["crash_recon"] = float(result.loss_recon)
            diagnostics["crash_kl"] = float(result.loss_kl)
            diagnostics["crash_kl_weight"] = float(result.kl_weight)
            diagnostics["saved_epoch"] = int(last_good["epoch"])
            diagnostics["saved_step"] = int(last_good["step"])
            diagnostics["saved_kl_weight"] = float(last_good["kl_weight"])

            save_failure_tensors(
                stats_dir=self.stats_dir,
                loss_policy=self.loss_policy,
                epoch=last_good["epoch"],
                step=last_good["step"],
                x_batch_train=last_good["x_batch_train"],
                mu=last_good["mu"],
                log_var=last_good["log_var"],
                curr_loss_recon=last_good["curr_loss_recon"],
                curr_loss_kl=last_good["curr_loss_kl"],
                diagnostics=diagnostics,
                file_tag="last_good_before_crash",
            )

            save_failure_history(
                stats_dir=self.stats_dir,
                recent_good_steps=self.recent_good_steps,
            )
        else:
            diagnostics["note"] = (
                "No earlier finite step available; saved crashing step"
            )
            save_failure_tensors(
                stats_dir=self.stats_dir,
                loss_policy=self.loss_policy,
                epoch=result.epoch,
                step=result.step,
                x_batch_train=result.x_batch_train,
                mu=result.mu,
                log_var=result.log_var,
                curr_loss_recon=result.loss_recon,
                curr_loss_kl=result.loss_kl,
                diagnostics=diagnostics,
                file_tag="crash_step",
            )

        exc = TrainingDivergedError(
            f"Training diverged at epoch={result.epoch}, step={result.step}, "
            f"recon={result.loss_recon}, kl={result.loss_kl}, "
            f"kl_weight={result.kl_weight}"
        )
        return GuardDecision(
            should_raise=True,
            exception=exc,
        )

    def _handle_tripwire_skip(self, result: StepResult) -> GuardDecision:
        self.consecutive_tripwires += 1

        old_lr = float(self.optimizer.learning_rate.numpy())
        new_lr = max(old_lr * self.lr_backoff, self.lr_floor)
        self.optimizer.learning_rate.assign(new_lr)

        diagnostics = latent_diagnostics(result.mu, result.log_var)
        diagnostics["note"] = (
            "Trip-wire fired; update skipped; accepted training state unchanged"
        )
        diagnostics["kl_jump_ratio"] = float(result.kl_jump_ratio)
        diagnostics["grad_norm"] = float(result.grad_norm)
        diagnostics["max_log_var"] = float(result.max_log_var)
        diagnostics["min_log_var"] = float(result.min_log_var)
        diagnostics["max_abs_mu"] = float(result.max_abs_mu)
        diagnostics["old_lr"] = old_lr
        diagnostics["new_lr"] = new_lr
        diagnostics["consecutive_tripwires"] = int(self.consecutive_tripwires)
        diagnostics["max_consecutive_tripwires"] = int(self.max_consecutive_tripwires)
        diagnostics["tripwire_tests_triggered"] = np.array(result.toxic_reasons, dtype=str)
        diagnostics["tripwire_threshold_kl_jump_ratio"] = float(self.kl_jump_ratio_threshold)
        diagnostics["tripwire_threshold_kl_abs"] = float(self.kl_abs_threshold)
        diagnostics["tripwire_threshold_max_log_var"] = float(self.max_log_var_threshold)

        logging.error(
            "Trip-wire fired at epoch=%d step=%d; skipped update; "
            "triggered_tests=%s recon=%g kl=%g kl_jump_ratio=%g max_log_var=%g "
            "min_log_var=%g max_abs_mu=%g grad_norm=%g "
            "kl_weight=%g lr %g -> %g "
            "consecutive_tripwires=%d/%d",
            result.epoch,
            result.step,
            result.toxic_reasons,
            result.loss_recon,
            result.loss_kl,
            result.kl_jump_ratio,
            result.max_log_var,
            result.min_log_var,
            result.max_abs_mu,
            result.grad_norm,
            result.kl_weight,
            old_lr,
            new_lr,
            self.consecutive_tripwires,
            self.max_consecutive_tripwires,
        )

        save_failure_tensors(
            stats_dir=self.stats_dir,
            loss_policy=self.loss_policy,
            epoch=result.epoch,
            step=result.step,
            x_batch_train=result.x_batch_train,
            mu=result.mu,
            log_var=result.log_var,
            curr_loss_recon=result.loss_recon,
            curr_loss_kl=result.loss_kl,
            diagnostics=diagnostics,
            file_tag="tripwire_skip",
        )

        if self.consecutive_tripwires >= self.max_consecutive_tripwires:
            exc = TrainingDivergedError(
                f"Stopping after {self.consecutive_tripwires} consecutive trip-wires "
                f"at epoch={result.epoch}, step={result.step}, "
                f"recon={result.loss_recon}, kl={result.loss_kl}, "
                f"kl_weight={result.kl_weight}"
            )
            return GuardDecision(
                should_raise=True,
                exception=exc,
            )

        return GuardDecision(
            should_continue=True,
        )