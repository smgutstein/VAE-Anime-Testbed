import json
import logging
from pathlib import Path
from time import ctime

import numpy as np
import tensorflow as tf


def _json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj

def latent_diagnostics(mu, log_var):
    sigma = tf.exp(0.5 * log_var)

    mu_finite = bool(tf.reduce_all(tf.math.is_finite(mu)).numpy())
    log_var_finite = bool(tf.reduce_all(tf.math.is_finite(log_var)).numpy())
    sigma_finite = bool(tf.reduce_all(tf.math.is_finite(sigma)).numpy())

    diagnostics = {
        "mu_finite": mu_finite,
        "log_var_finite": log_var_finite,
        "sigma_finite": sigma_finite,
        "mu_min": float(tf.reduce_min(mu).numpy()),
        "mu_max": float(tf.reduce_max(mu).numpy()),
        "mu_mean": float(tf.reduce_mean(mu).numpy()),
        "log_var_min": float(tf.reduce_min(log_var).numpy()),
        "log_var_max": float(tf.reduce_max(log_var).numpy()),
        "log_var_mean": float(tf.reduce_mean(log_var).numpy()),
        "sigma_min": float(tf.reduce_min(sigma).numpy()),
        "sigma_max": float(tf.reduce_max(sigma).numpy()),
        "sigma_mean": float(tf.reduce_mean(sigma).numpy()),
    }
    return diagnostics


def save_failure_tensors(
    stats_dir,
    loss_policy,
    epoch,
    step,
    x_batch_train,
    mu,
    log_var,
    curr_loss_recon,
    curr_loss_kl,
    diagnostics=None,
    max_items=8,
    file_tag="failure",
):
    x_np = x_batch_train.numpy()[:max_items]

    mu_full = mu.numpy()
    log_var_full = log_var.numpy()
    sigma = np.exp(np.clip(0.5 * log_var_full, -50.0, 50.0))

    mu_np = mu_full[:max_items]
    log_var_np = log_var_full[:max_items]
    sigma_np = sigma[:max_items]
    flat_idx = np.argsort(log_var_full.ravel())[-20:]

    out_path = stats_dir / Path(f"{file_tag}_epoch{epoch}_step{step}.npz")

    payload = {
        "epoch": np.array(epoch, dtype=np.int32),
        "step": np.array(step, dtype=np.int32),
        "loss_recon": np.array(curr_loss_recon, dtype=np.float32),
        "loss_kl": np.array(curr_loss_kl, dtype=np.float32),
        "kl_weight": np.array(loss_policy.current_value(), dtype=np.float32),
        "x_batch_train": x_np,
        "mu": mu_np,
        "log_var": log_var_np,
        "sigma": sigma_np,
        "top20_log_var_values": log_var_full.ravel()[flat_idx],
        "top20_sigma_values": sigma.ravel()[flat_idx],
        "top20_flat_indices": flat_idx,
        "mu_is_finite": np.array(np.isfinite(mu_full).all(), dtype=np.bool_),
        "log_var_is_finite": np.array(np.isfinite(log_var_full).all(), dtype=np.bool_),
        "sigma_is_finite": np.array(np.isfinite(sigma).all(), dtype=np.bool_),
    }

    if diagnostics is not None:
        with open(stats_dir / Path(f"diagnostic_epoch{epoch}_step{step}.json"), "w") as f:
            json.dump(_json_safe(diagnostics), f, indent=2)

    np.savez_compressed(out_path, **payload)
    logging.error("Saved failure tensors to %s", out_path)

def save_failure_history(stats_dir, recent_good_steps, max_items=8):
    """
    Save the rolling buffer of recent finite steps so the lead-up to
    divergence can be inspected after the run fails.
    """
    history_payload = {}

    for i, item in enumerate(recent_good_steps):
        prefix = f"item_{i:02d}"
        x_np = item["x_batch_train"].numpy()[:max_items]
        mu_np = item["mu"].numpy()[:max_items]
        log_var_np = item["log_var"].numpy()[:max_items]
        sigma_np = np.exp(np.clip(0.5 * log_var_np, -50.0, 50.0))

        history_payload[f"{prefix}_epoch"] = np.array(item["epoch"], dtype=np.int32)
        history_payload[f"{prefix}_step"] = np.array(item["step"], dtype=np.int32)
        history_payload[f"{prefix}_loss_recon"] = np.array(item["curr_loss_recon"], dtype=np.float32)
        history_payload[f"{prefix}_loss_kl"] = np.array(item["curr_loss_kl"], dtype=np.float32)
        history_payload[f"{prefix}_kl_weight"] = np.array(item["kl_weight"], dtype=np.float32)
        history_payload[f"{prefix}_x_batch_train"] = x_np
        history_payload[f"{prefix}_mu"] = mu_np
        history_payload[f"{prefix}_log_var"] = log_var_np
        history_payload[f"{prefix}_sigma"] = sigma_np

    out_path = stats_dir / Path("precrash_history.npz")
    np.savez_compressed(out_path, **history_payload)
    logging.error("Saved pre-crash history to %s", out_path)

def write_run_summary(
    output_dir,
    cfg,
    loss_policy,
    status,
    start_time,
    end_time,
    final_recon=None,
    final_kl=None,
    final_kl_weight=None,
    error_message=None,
):
    summary_path = output_dir / "run_summary.json"

    runtime_seconds = None
    if start_time is not None and end_time is not None:
        runtime_seconds = float(end_time - start_time)

    summary = {
        "experiment_dir": str(output_dir),
        "experiment_name": cfg.expt_name,
        "status": status,
        "error": error_message,
        "config_file": str(cfg.config_file),
        "loss_policy": loss_policy.policy_name,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "latent_dim": cfg.latent_dim,
        "batch_size": cfg.batch_size,
        "image_size": cfg.image_size,
        "save_net": cfg.save_net,
        "beta": cfg.beta,
        "beta_eff": cfg.beta/cfg.latent_dim if cfg.beta is not None else None,
        "initial_kl_weight": cfg.initial_kl_weight,
        "max_kl_weight": cfg.max_kl_weight,
        "kl_weight_update_factor": cfg.kl_weight_update_factor,
        "running_window": cfg.running_window,
        "final_recon_loss": final_recon,
        "final_kl_loss": final_kl,
        "final_kl_weight": final_kl_weight,
        "start_time": ctime(start_time) if start_time is not None else None,
        "end_time": ctime(end_time) if end_time is not None else None,
        "runtime_seconds": runtime_seconds,
    }

    summary = {k: _json_safe(v) for k, v in summary.items()}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Wrote run summary to %s", summary_path)