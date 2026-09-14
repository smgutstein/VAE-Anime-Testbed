"""Restart-capable epoch-boundary checkpoints for VAE training."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from VAE_Anime_Artifacts import (
    LATENT_KL_STATS_FILE,
    LATENT_STATS_FILE,
    LATENT_VAR_STATS_FILE,
    LOSS_EVENTS_FILE,
    LOSS_TEXT_FILE,
    VAL_LATENT_STATS_FILE,
    VAL_LOSS_TEXT_FILE,
)


STATE_FILE = "resume_state.json"
STATE_PREFIX = "training_state"
FULL_CHECKPOINT_RE = re.compile(r"^epoch_(?P<epoch>\d+)_step_(?P<step>\d+)$")
ARTIFACT_NAMES = (
    LOSS_TEXT_FILE,
    VAL_LOSS_TEXT_FILE,
    LOSS_EVENTS_FILE,
    LATENT_STATS_FILE,
    LATENT_VAR_STATS_FILE,
    LATENT_KL_STATS_FILE,
    VAL_LATENT_STATS_FILE,
)


def _checkpoint_root(output_dir):
    return Path(output_dir) / "checkpoints" / "training_state"


def save_training_checkpoint(
    *, output_dir, vae_net, optimizer, prev_kl_tensor, beta_factor,
    loss_policy, step_guard, epoch, step,
):
    """Save TensorFlow and Python-side state after a completed epoch."""
    checkpoint_dir = _checkpoint_root(output_dir) / f"epoch_{epoch:04d}_step_{step:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = tf.train.Checkpoint(
        vae=vae_net,
        optimizer=optimizer,
        prev_kl=prev_kl_tensor,
        beta_factor=beta_factor,
    )
    checkpoint.write(str(checkpoint_dir / STATE_PREFIX))

    stats_dir = Path(output_dir) / "stats"
    artifact_sizes = {
        name: (stats_dir / name).stat().st_size
        for name in ARTIFACT_NAMES
        if (stats_dir / name).is_file()
    }
    payload = {
        "schema_version": 1,
        "epoch": int(epoch),
        "step": int(step),
        "next_epoch": int(epoch) + 1,
        "loss_policy": loss_policy.get_state(),
        "step_guard": step_guard.get_state(),
        "artifact_sizes": artifact_sizes,
        "randomness_note": (
            "Model, optimizer, loss-policy, and StepGuard state are restored. "
            "Python/NumPy/TensorFlow random streams and tf.data shuffle position "
            "are recreated, so continuation is not promised to be bit-for-bit identical."
        ),
    }
    tmp_path = checkpoint_dir / f"{STATE_FILE}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(checkpoint_dir / STATE_FILE)
    logging.info("Saved restart-capable checkpoint to %s", checkpoint_dir)
    return checkpoint_dir


def find_latest_training_checkpoint(output_dir):
    root = _checkpoint_root(output_dir)
    candidates = []
    if root.is_dir():
        for path in root.iterdir():
            match = FULL_CHECKPOINT_RE.match(path.name)
            if match and is_complete_training_checkpoint(path):
                candidates.append((int(match.group("epoch")), int(match.group("step")), path))
    if not candidates:
        raise FileNotFoundError(f"No restart-capable checkpoint found under {root}")
    return max(candidates)[2]


def is_complete_training_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    return (
        checkpoint_dir.is_dir()
        and (checkpoint_dir / STATE_FILE).is_file()
        and (checkpoint_dir / f"{STATE_PREFIX}.index").is_file()
        and any(checkpoint_dir.glob(f"{STATE_PREFIX}.data-*-of-*"))
    )


def resolve_training_checkpoint(output_dir, requested=None):
    """Resolve a full-state checkpoint, defaulting to the latest complete one."""
    output_dir = Path(output_dir)
    if requested is None:
        return find_latest_training_checkpoint(output_dir)

    requested = Path(requested)
    if requested.name.endswith(".weights.h5"):
        raise ValueError(
            "Weight-only files are not restart checkpoints; select a full-state "
            "checkpoint directory"
        )

    candidates = [requested] if requested.is_absolute() else [
        output_dir / "checkpoints" / "training_state" / requested,
        output_dir / "checkpoints" / requested,
        requested,
    ]
    checkpoint_dir = next((path for path in candidates if path.exists()), candidates[0])
    if not is_complete_training_checkpoint(checkpoint_dir):
        raise FileNotFoundError(
            f"Complete restart checkpoint does not exist: {checkpoint_dir}"
        )
    return checkpoint_dir


def load_resume_metadata(checkpoint_dir):
    path = Path(checkpoint_dir) / STATE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Restart metadata does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported restart checkpoint schema in {path}")
    return payload


def preserve_and_rollback_artifacts(output_dir, artifact_sizes, epoch):
    """Back up and truncate artifact tails written after the checkpoint."""
    stats_dir = Path(output_dir) / "stats"
    oversized = []
    for name, saved_size in artifact_sizes.items():
        path = stats_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Checkpointed artifact is missing: {path}")
        current_size = path.stat().st_size
        if current_size < saved_size:
            raise ValueError(
                f"{path} is smaller than its checkpointed size "
                f"({current_size} < {saved_size})"
            )
        if current_size > saved_size:
            oversized.append((path, int(saved_size)))

    if not oversized:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(output_dir) / "resume_discarded_tails" / f"after_epoch_{epoch:04d}_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path, saved_size in oversized:
        with path.open("rb") as source:
            source.seek(saved_size)
            tail = source.read()
        (backup_dir / f"{path.name}.tail").write_bytes(tail)
        with path.open("r+b") as target:
            target.truncate(saved_size)
    logging.info("Preserved post-checkpoint artifact tails in %s", backup_dir)
    return backup_dir


def restore_training_checkpoint(
    *, checkpoint_dir, vae_net, optimizer, prev_kl_tensor, beta_factor,
    loss_policy, step_guard,
):
    checkpoint_dir = Path(checkpoint_dir)
    payload = load_resume_metadata(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(
        vae=vae_net,
        optimizer=optimizer,
        prev_kl=prev_kl_tensor,
        beta_factor=beta_factor,
    )
    status = checkpoint.read(str(checkpoint_dir / STATE_PREFIX))
    status.assert_existing_objects_matched()
    loss_policy.set_state(payload["loss_policy"])
    step_guard.set_state(payload["step_guard"])
    return payload

