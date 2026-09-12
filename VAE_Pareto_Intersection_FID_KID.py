"""
Recover VAEs near a Pareto-front intersection and evaluate image-distribution
quality with a fixed reference VAE.

WHY THIS MODULE EXISTS
----------------------
The training runs save:
    * the full chronological recon/KL history;
    * periodic VAE weight checkpoints;
    * one separately trained "reference VAE" chosen by validation SSIM.

The exact Pareto-curve intersection between two experiments will usually NOT
occur at exactly the same epoch as a periodic checkpoint.  Therefore this
module separates the problem into two stages:

    1. Find the desired point in Pareto space using the complete loss history.
    2. Among the VAEs that were actually saved, recover the checkpoint whose
       recon/KL coordinates are closest to that Pareto target.

The selected checkpoint is then reconstructed from:
    * expts/expt_N/config.ini          -> architecture/configuration
    * expts/expt_N/checkpoints/*.h5   -> learned weights

The checkpoint is only the STARTING POINT.  The module then rederives a VAE
at the intersection-side epoch/step by continuing training from that earlier
checkpoint before calculating image-distribution metrics.

IMPORTANT: periodic checkpoints contain model weights only.  Optimizer state,
TensorFlow/Python/NumPy RNG state, tf.data iterator state, and adaptive-controller
state were not saved.  Therefore the rederived VAE is an approximation to the
original VAE that existed at the historical intersection point; it cannot be a
bit-for-bit recovery of that original state.

REFERENCE FEATURE SPACE
-----------------------
Standard FID/KID use features from a pretrained Inception network.  This
project instead uses a fixed reference VAE encoder as an anime-specific feature
extractor.  Every raw, reconstructed, and generated image is passed through the
SAME frozen reference encoder.

Therefore the metrics reported here are deliberately named:

    vae_feature_fid
    vae_feature_kid

They should NOT be described as standard Inception FID/KID.

DEFAULT FEATURE LAYER
---------------------
By default the module uses encoder layer "lrelu_4", the final deterministic
hidden representation immediately before latent_mu / latent_log_var in the
current architecture.  The layer can be changed from the command line with
--feature_layer.

IMAGE SETS COMPARED
-------------------
For each recovered candidate VAE, the module calculates three distribution
comparisons:

    generated_vs_raw
        Prior samples z ~ N(0,I) decoded by the candidate VAE, compared with
        real validation images.  This is the primary generation-quality result.

    reconstructed_vs_raw
        Deterministic reconstructions decoder(mu(x)) compared with the original
        validation images.  This measures how much the candidate reconstruction
        pipeline alters the real-image distribution.

    generated_vs_reconstructed
        Prior-generated images compared with deterministic reconstructions.
        This is mainly diagnostic: it measures the gap between what the decoder
        produces from prior samples and what it produces from data-derived
        latent codes.

FAIR-COMPARISON RULES
---------------------
To avoid adding sampling noise to experiment A vs experiment B:

    * both candidate VAEs use the same raw validation images;
    * both candidate VAEs use the same fixed prior latent vectors;
    * all images use the same fixed reference encoder;
    * KID subset selection is seeded.

PARETO GEOMETRY
---------------
The project's existing intersection code works in normalized coordinates:

    x = normalized reconstruction loss
    y = normalized log10(KL loss)

This module reuses exactly that geometry when choosing the nearest saved
checkpoint.  It does NOT simply choose the checkpoint nearest in epoch.

HIGH-LEVEL WORKFLOW
-------------------
1. Load the two experiments' chronological loss histories.
2. Compute the post-burn-in Pareto fronts using existing project utilities.
3. Find exact Pareto intersection(s), or the closest approach if none exists.
4. Match every periodic checkpoint to its recorded recon/KL point.
5. Choose the nearest EARLIER checkpoint from each experiment.
6. Rebuild both candidate VAEs and load those checkpoint weights.
7. Continue training each checkpoint through its intersection-side epoch/step,
   using the original run's recorded KL weight at each resumed step.
8. Save the two rederived intersection VAE weights.
9. Rebuild the saved reference VAE and expose one encoder hidden layer.
10. Recreate one fixed validation image set.
11. Generate one fixed set of prior latent vectors.
12. Produce raw/reconstructed/generated feature vectors.
13. Calculate VAE-feature Fréchet distance and KID.
14. Save both JSON and human-readable text results.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from time import perf_counter

import numpy as np

# Match the training script's behavior of hiding TensorFlow's noisy C++ runtime
# messages.  Level 2 suppresses INFO and WARNING messages (including the benign
# finite-dataset OUT_OF_RANGE warning) while still leaving genuine ERROR logs
# visible.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from VAE_Anime_Config import TrainerConfig
from VAE_Anime_Datasets import Datasets
from VAE_Anime_Full_Model import VAE_Model
from VAE_Loss_Records import read_loss_records, pareto_records
from VAE_Pareto_Front_Intersections import (
    closest_curve_approach,
    exact_curve_intersections,
    make_normalizer,
    midpoint,
    normalized_curve,
    transformed_point,
)


# Periodic checkpoints created by the current trainer use filenames such as:
#     epoch_0050_step_0032.weights.h5
# The epoch and step encoded in the filename let us match a saved VAE exactly
# back to the corresponding record in losses_file.txt.
CHECKPOINT_RE = re.compile(
    r"^epoch_(?P<epoch>\d+)_step_(?P<step>\d+)\.weights\.h5$"
)


# ---------------------------------------------------------------------------
# Experiment / Pareto / checkpoint discovery
# ---------------------------------------------------------------------------

def resolve_experiment_dir(parent_dir: Path, expt_num: int) -> Path:
    """Return expts/expt_N and fail early if the experiment is missing."""

    path = Path(parent_dir) / f"expt_{expt_num}"
    if not path.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {path}")
    return path


def retained_pareto_records(expt_dir: Path, skip_fraction: float) -> list[dict]:
    """
    Load one experiment's loss history and return its retained Pareto frontier.

    The skip_fraction behavior deliberately matches the existing project
    analysis: a chronological burn-in fraction is removed BEFORE computing the
    frontier.  Keeping this definition identical matters because otherwise the
    intersection used for VAE recovery could differ from the plotted frontier.
    """

    records = read_loss_records(expt_dir)

    # Burn-in is chronological, not a percentage of Pareto points.
    skip_points = int(skip_fraction * len(records))
    retained = records[skip_points:]
    if len(retained) < 2:
        raise ValueError(
            f"{expt_dir.name}: too few records after skip_fraction={skip_fraction}"
        )
    frontier = pareto_records(retained)
    if len(frontier) < 2:
        raise ValueError(f"{expt_dir.name}: Pareto frontier has fewer than 2 points")
    return frontier


def find_checkpoint_records(expt_dir: Path) -> list[dict]:
    """
    Match every periodic weights file to its exact loss-history record.

    The weights file itself contains no recon/KL coordinates.  We recover those
    by parsing epoch/step from the filename and joining on the same epoch/step
    in losses_file.txt.

    The returned dictionaries therefore contain BOTH:
        * the recorded training state (recon, KL, iteration, etc.);
        * checkpoint_path, which can be loaded to reconstruct that VAE.
    """

    checkpoint_dir = expt_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # Create an exact lookup from training state -> loss record.
    # Epoch/step is used rather than iteration because it is encoded directly
    # in the checkpoint filename.
    records_by_key = {
        (r["epoch"], r["step"]): r
        for r in read_loss_records(expt_dir)
    }

    result = []
    unmatched = []

    for path in sorted(checkpoint_dir.iterdir()):
        if not path.is_file():
            continue
        match = CHECKPOINT_RE.match(path.name)
        if match is None:
            continue

        key = (int(match.group("epoch")), int(match.group("step")))
        loss_record = records_by_key.get(key)
        if loss_record is None:
            unmatched.append(path.name)
            continue

        result.append({
            **loss_record,
            "checkpoint_path": path,
        })

    if unmatched:
        print(
            f"Warning: {expt_dir.name}: {len(unmatched)} checkpoint(s) could not "
            "be matched exactly to an epoch/step in losses_file.txt"
        )

    if not result:
        raise ValueError(
            f"{expt_dir.name}: no periodic checkpoints could be matched to loss records"
        )

    return result


def normalized_distance(record: dict, target, normalize_xy) -> float:
    """
    Distance from one saved checkpoint to the chosen Pareto target.

    transformed_point() applies the project's recon/log10(KL) transform.
    normalize_xy() then places recon and KL on comparable numerical scales.
    Without this normalization, whichever axis has the larger numeric range
    would dominate checkpoint selection.
    """

    point = normalize_xy(*transformed_point(record))
    dx = point[0] - target[0]
    dy = point[1] - target[1]
    return math.hypot(dx, dy)


def nearest_checkpoint(
    checkpoints,
    target,
    normalize_xy,
    *,
    max_iteration=None,
):
    """
    Return the latest saved checkpoint at or before the target iteration.

    If max_iteration is supplied, only checkpoints from that iteration or
    earlier are eligible.  For rederivation, chronological closeness is what
    matters: the latest eligible checkpoint minimizes the amount of training
    that must be replayed before reaching the intersection-side state.
    """

    eligible = checkpoints

    if max_iteration is not None:
        eligible = [
            record
            for record in checkpoints
            if record["iteration"] <= max_iteration
        ]

        if not eligible:
            raise ValueError(
                "No saved checkpoint exists at or before "
                f"intersection iteration {max_iteration}"
            )

    best = max(
        eligible,
        key=lambda r: r["iteration"],
    )
    return best, normalized_distance(best, target, normalize_xy)


def nearest_frontier_record(frontier, target, normalize_xy):
    """
    Return the actual Pareto record nearest to a geometric intersection target.

    An exact curve intersection can lie between two recorded Pareto points, so
    it does not necessarily have an epoch/step/iteration of its own.  We use the
    nearest real frontier record from each experiment as that experiment's
    chronological reference point, then require the recovered checkpoint to be
    from that iteration or earlier.
    """

    record = min(
        frontier,
        key=lambda r: normalized_distance(r, target, normalize_xy),
    )
    return record


def choose_target(frontier_a, frontier_b, checkpoints_a, checkpoints_b):
    """
    Choose the Pareto comparison target and the best recoverable VAE pair.

    Cases
    -----
    1. Exact intersection(s) exist:
       If there is only one, use it.  If there are several, choose the exact
       intersection for which the AVAILABLE periodic checkpoints jointly give
       the closest recoverable pair.

    2. No exact intersection exists:
       Use the midpoint of the project's closest-approach pair as the target.

    In both cases, checkpoint matching is done in the same normalized
    (reconstruction, log10(KL)) geometry used by the intersection code.
    """

    # Build one common coordinate transform from BOTH frontiers so experiment A
    # and B are measured on exactly the same normalized axes.
    normalize_xy, denormalize_xy = make_normalizer(frontier_a, frontier_b)
    curve_a = normalized_curve(frontier_a, normalize_xy)
    curve_b = normalized_curve(frontier_b, normalize_xy)

    intersections = exact_curve_intersections(curve_a, curve_b)

    if intersections:
        # If there are several exact intersections, choose the one best
        # represented by the available periodic checkpoints.
        candidates = []
        for item in intersections:
            target = item["point"]

            # The geometric intersection may fall between recorded frontier
            # points.  Find the nearest REAL Pareto point in each experiment so
            # we have an experiment-specific iteration cutoff.
            frontier_record_a = nearest_frontier_record(
                frontier_a,
                target,
                normalize_xy,
            )
            frontier_record_b = nearest_frontier_record(
                frontier_b,
                target,
                normalize_xy,
            )

            # Recover only checkpoints from at or before the intersection
            # location for that experiment.
            cp_a, dist_a = nearest_checkpoint(
                checkpoints_a,
                target,
                normalize_xy,
                max_iteration=frontier_record_a["iteration"],
            )
            cp_b, dist_b = nearest_checkpoint(
                checkpoints_b,
                target,
                normalize_xy,
                max_iteration=frontier_record_b["iteration"],
            )

            candidates.append({
                "kind": "exact_intersection",
                "target": target,
                "checkpoint_a": cp_a,
                "checkpoint_b": cp_b,
                "distance_a": dist_a,
                "distance_b": dist_b,
                "combined_distance": math.hypot(dist_a, dist_b),
                "intersection_count": len(intersections),
                "frontier_record_a": frontier_record_a,
                "frontier_record_b": frontier_record_b,
            })

        chosen = min(candidates, key=lambda x: x["combined_distance"])

    else:
        closest = closest_curve_approach(curve_a, curve_b)
        target = midpoint(closest["point_a"], closest["point_b"])

        frontier_record_a = nearest_frontier_record(
            frontier_a,
            closest["point_a"],
            normalize_xy,
        )
        frontier_record_b = nearest_frontier_record(
            frontier_b,
            closest["point_b"],
            normalize_xy,
        )

        cp_a, dist_a = nearest_checkpoint(
            checkpoints_a,
            target,
            normalize_xy,
            max_iteration=frontier_record_a["iteration"],
        )
        cp_b, dist_b = nearest_checkpoint(
            checkpoints_b,
            target,
            normalize_xy,
            max_iteration=frontier_record_b["iteration"],
        )

        chosen = {
            "kind": "closest_approach",
            "target": target,
            "checkpoint_a": cp_a,
            "checkpoint_b": cp_b,
            "distance_a": dist_a,
            "distance_b": dist_b,
            "combined_distance": math.hypot(dist_a, dist_b),
            "intersection_count": 0,
            "curve_distance": math.sqrt(closest["distance_sq"]),
            "frontier_record_a": frontier_record_a,
            "frontier_record_b": frontier_record_b,
        }

    # Convert the normalized target back to physically interpretable values
    # for logging / JSON output.  The y-coordinate is log10(KL), so exponentiate.
    raw_recon, raw_log_kl = denormalize_xy(*chosen["target"])
    chosen["target_recon_loss"] = raw_recon
    chosen["target_kl_loss"] = 10.0 ** raw_log_kl
    chosen["normalize_xy"] = normalize_xy
    return chosen


# ---------------------------------------------------------------------------
# Rebuilding candidate and reference VAEs
# ---------------------------------------------------------------------------

def build_vae_from_config(config_path: Path, output_dir: Path) -> tuple[VAE_Model, TrainerConfig]:
    """
    Recreate the VAE architecture from a saved experiment config.

    Periodic checkpoints contain weights only.  The experiment's config.ini is
    therefore the architecture specification needed before load_weights().
    """

    cfg = TrainerConfig.from_file(config_path)
    vae = VAE_Model(
        enc_input_shape=(cfg.image_size, cfg.image_size, 3),
        latent_dim=cfg.latent_dim,
        base_filters=cfg.base_filters,
        filter_factors=cfg.filter_factors,
        encode_dense_units=cfg.encode_dense_units,
        kernel_size=cfg.kernel_size,
        output_dir=output_dir,
    )
    return vae, cfg


def load_experiment_checkpoint(expt_dir: Path, checkpoint: dict) -> tuple[VAE_Model, TrainerConfig]:
    """Rebuild one candidate VAE and load the selected periodic weights."""

    vae, cfg = build_vae_from_config(
        expt_dir / "config.ini",
        expt_dir / "model_info",
    )
    vae.vae_net.load_weights(checkpoint["checkpoint_path"])
    return vae, cfg



def make_candidate_training_dataset(cfg: TrainerConfig, scratch_dir: Path):
    """
    Recreate the candidate experiment's training dataset.

    This uses the experiment's saved config and seed.  Train/validation
    membership therefore follows the project's current dataset logic.

    NOTE:
        The periodic checkpoint does not preserve the tf.data iterator/RNG
        position from the historical run.  The sequence of batches seen during
        this short rederivation may therefore differ from the original run.
    """

    data = Datasets(
        output_dir=scratch_dir,
        seed=cfg.seed,
        data_dir=cfg.data_dir,
        strict_reproducibility=True,
    )
    data.set_data_params(
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        val_split=cfg.val_split,
        shuffle_buffer=cfg.shuffle_buffer,
        train_drop_remainder=cfg.train_drop_remainder,
    )
    data.download_data()
    data.make_train_and_validation_sets()
    return data.training_dataset


def rederivation_train_step(
    x_batch_train,
    kl_weight_tensor,
    vae_obj,
    loss_fn,
    optimizer,
):
    """
    Perform one VAE gradient update using the project's core training objective.

    The KL weight is supplied explicitly from the ORIGINAL experiment's
    losses_file.txt for this epoch/step.  This is deliberate: for an adaptive
    run, the original controller state was not saved, so replaying the recorded
    KL-weight schedule is closer to the historical run than restarting a fresh
    controller at the checkpoint.
    """

    model = vae_obj.vae_net

    with tf.GradientTape() as tape:
        reconstructed, mu, log_var = model(x_batch_train)

        loss_recon = (
            loss_fn(x_batch_train, reconstructed)
            * vae_obj.encoder.num_input_pixels
        )

        loss_kl = (
            tf.reduce_mean(
                1 + log_var - tf.square(mu) - tf.exp(log_var)
            )
            * -0.5
        )

        loss_tot = loss_recon + kl_weight_tensor * loss_kl

    grads = tape.gradient(loss_tot, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_recon, loss_kl


def rederive_to_intersection(
    *,
    vae: VAE_Model,
    cfg: TrainerConfig,
    expt_dir: Path,
    checkpoint: dict,
    target_record: dict,
    scratch_dir: Path,
):
    """
    Continue training an earlier periodic checkpoint to the intersection-side
    epoch/step.

    Returns
    -------
    dict
        Metadata describing how much rederivation was performed and the losses
        reached on the final rederived training step.

    Method
    ------
    * Start from checkpoint weights already loaded into ``vae``.
    * Create a fresh Adam optimizer using the original configured learning rate.
      Optimizer state from the historical run is unavailable.
    * Recreate the training dataset from the experiment config.
    * For every resumed epoch/step, use the ORIGINAL recorded kl_weight from
      losses_file.txt.
    * Stop immediately after target_record['epoch']/['step'].

    Because optimizer/RNG/data-iterator state was not checkpointed, this creates
    a model at the same nominal training location, but not the exact historical
    model that existed there.
    """

    cp_iteration = int(checkpoint["iteration"])
    target_iteration = int(target_record["iteration"])

    if cp_iteration > target_iteration:
        raise ValueError(
            f"{expt_dir.name}: checkpoint iteration {cp_iteration} is after "
            f"target iteration {target_iteration}"
        )

    # No training is needed if a checkpoint happens to land exactly on target.
    if cp_iteration == target_iteration:
        return {
            "steps_rederived": 0,
            "start_epoch": int(checkpoint["epoch"]),
            "start_step": int(checkpoint["step"]),
            "target_epoch": int(target_record["epoch"]),
            "target_step": int(target_record["step"]),
            "target_iteration": target_iteration,
            "final_rederived_recon_loss": float(checkpoint["recon_loss"]),
            "final_rederived_kl_loss": float(checkpoint["kl_loss"]),
            "note": "Checkpoint already coincides with target iteration.",
        }

    records = read_loss_records(expt_dir)
    records_by_key = {
        (int(r["epoch"]), int(r["step"])): r
        for r in records
    }

    # Determine whether the checkpoint is an end-of-epoch checkpoint.
    max_step_by_epoch = {}
    for record in records:
        epoch = int(record["epoch"])
        step = int(record["step"])
        max_step_by_epoch[epoch] = max(
            max_step_by_epoch.get(epoch, -1),
            step,
        )

    cp_epoch = int(checkpoint["epoch"])
    cp_step = int(checkpoint["step"])
    target_epoch = int(target_record["epoch"])
    target_step = int(target_record["step"])

    training_dataset = make_candidate_training_dataset(
        cfg,
        scratch_dir=scratch_dir,
    )

    # Periodic checkpoints save VAE weights only, so optimizer moments cannot
    # be restored.  Start a fresh optimizer with the original learning rate.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.learning_rate
    )
    loss_fn = tf.keras.losses.MeanSquaredError()

    steps_rederived = 0
    final_recon = None
    final_kl = None

    # If the checkpoint is at the last step of its epoch, the first update is
    # in the following epoch.  Otherwise resume later in that same epoch.
    if cp_step >= max_step_by_epoch.get(cp_epoch, cp_step):
        first_epoch = cp_epoch + 1
    else:
        first_epoch = cp_epoch

    print(
        f"{expt_dir.name}: rederiving from epoch {cp_epoch} step {cp_step} "
        f"to epoch {target_epoch} step {target_step}"
    )

    for epoch in range(first_epoch, target_epoch + 1):
        for step, x_batch_train in enumerate(training_dataset):
            # If resuming inside the checkpoint epoch, skip batches already
            # represented by the saved checkpoint.
            if epoch == cp_epoch and step <= cp_step:
                continue

            # On the final epoch, do not train past the target step.
            if epoch == target_epoch and step > target_step:
                break

            original = records_by_key.get((epoch, step))
            if original is None:
                raise ValueError(
                    f"{expt_dir.name}: no loss record for epoch {epoch}, "
                    f"step {step}; cannot recover original KL-weight schedule"
                )

            kl_weight_tensor = tf.convert_to_tensor(
                float(original["kl_weight"]),
                dtype=tf.float32,
            )

            loss_recon, loss_kl = rederivation_train_step(
                x_batch_train,
                kl_weight_tensor,
                vae,
                loss_fn,
                optimizer,
            )

            steps_rederived += 1
            final_recon = float(loss_recon.numpy())
            final_kl = float(loss_kl.numpy())

            if epoch == target_epoch and step == target_step:
                break

    if final_recon is None or final_kl is None:
        raise RuntimeError(
            f"{expt_dir.name}: rederivation performed no gradient updates "
            "despite checkpoint being earlier than target"
        )

    return {
        "steps_rederived": steps_rederived,
        "start_epoch": cp_epoch,
        "start_step": cp_step,
        "start_iteration": cp_iteration,
        "target_epoch": target_epoch,
        "target_step": target_step,
        "target_iteration": target_iteration,
        "original_target_recon_loss": float(target_record["recon_loss"]),
        "original_target_kl_loss": float(target_record["kl_loss"]),
        "final_rederived_recon_loss": final_recon,
        "final_rederived_kl_loss": final_kl,
        "note": (
            "Approximate rederivation: model weights were restored, but "
            "optimizer/RNG/tf.data iterator/controller state were not available."
        ),
    }


def load_reference_vae(ref_root: Path, ref_expt: int) -> tuple[VAE_Model, TrainerConfig, Path]:
    """
    Rebuild the fixed reference VAE from ref_vae/expt_N.

    This is independent of the two candidate experiments.  Once chosen, this
    same reference should be reused across experiments so all metric values live
    in one common feature space.
    """

    ref_dir = Path(ref_root) / f"expt_{ref_expt}"
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference VAE directory does not exist: {ref_dir}")

    weights_path = ref_dir / "reference_vae.weights.h5"
    config_path = ref_dir / "config.ini"

    if not weights_path.is_file():
        raise FileNotFoundError(f"Reference weights do not exist: {weights_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Reference config does not exist: {config_path}")

    vae, cfg = build_vae_from_config(config_path, ref_dir)
    vae.vae_net.load_weights(weights_path)
    return vae, cfg, ref_dir


# ---------------------------------------------------------------------------
# Building the fixed image-feature extractor
# ---------------------------------------------------------------------------

def make_reference_feature_model(ref_vae: VAE_Model, feature_layer: str):
    """
    Create a read-only Keras model exposing one reference-encoder hidden layer.

    We do NOT use the reference decoder here.  The reference VAE serves only as
    a fixed mapping:

        image -> feature vector

    Every raw/reconstructed/generated image from every candidate experiment is
    evaluated through this exact same mapping.
    """

    encoder = ref_vae.encoder.encoder_net
    try:
        layer = encoder.get_layer(feature_layer)
    except ValueError as exc:
        available = ", ".join(layer.name for layer in encoder.layers)
        raise ValueError(
            f"Reference encoder has no layer '{feature_layer}'. "
            f"Available layers: {available}"
        ) from exc

    # Reuse the already-trained encoder graph but terminate it at the selected
    # hidden layer rather than at latent_mu / latent_log_var.
    return tf.keras.Model(
        inputs=encoder.input,
        outputs=layer.output,
        name=f"reference_features_{feature_layer}",
    )


# ---------------------------------------------------------------------------
# Building one fixed evaluation image set
# ---------------------------------------------------------------------------

def make_reference_validation_dataset(cfg: TrainerConfig, scratch_dir: Path):
    """
    Recreate the reference VAE's validation split.

    Using the reference config/seed rather than either candidate experiment's
    config makes the raw comparison set independent of experiment A vs B.
    """

    data = Datasets(
        output_dir=scratch_dir,
        seed=cfg.seed,
        data_dir=cfg.data_dir,
        strict_reproducibility=True,
    )
    data.set_data_params(
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        val_split=cfg.val_split,
        shuffle_buffer=cfg.shuffle_buffer,
        train_drop_remainder=cfg.train_drop_remainder,
    )
    data.download_data()
    data.make_train_and_validation_sets()
    return data.validation_dataset


def collect_images(dataset, n_images: int) -> np.ndarray:
    """Materialize exactly n_images from a tf.data dataset into one array."""

    batches = []
    count = 0
    for batch in dataset:
        array = np.asarray(batch)
        take = min(len(array), n_images - count)
        if take > 0:
            batches.append(array[:take])
            count += take
        if count >= n_images:
            break

    if count < n_images:
        raise ValueError(
            f"Requested {n_images} raw images but dataset yielded only {count}"
        )

    return np.concatenate(batches, axis=0)


# ---------------------------------------------------------------------------
# Producing reconstructed / generated image sets
# ---------------------------------------------------------------------------

def reconstruct_mu(vae: VAE_Model, raw_images: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Deterministically reconstruct images via decoder(mu(x)).

    We deliberately do NOT use the sampled latent z here.  Using mu removes
    posterior-sampling noise, which makes experiment-to-experiment reconstruction
    comparisons cleaner.
    """

    outputs = []
    for start in range(0, len(raw_images), batch_size):
        batch = tf.convert_to_tensor(raw_images[start:start + batch_size])
        mu, _, _ = vae.encoder.encoder_net(batch, training=False)
        recon = vae.decoder.decoder_net(mu, training=False)
        outputs.append(np.asarray(recon))
    return np.concatenate(outputs, axis=0)


def generate_images(
    vae: VAE_Model,
    latent_samples: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """
    Decode a pre-generated, fixed array of prior samples.

    The caller creates latent_samples once and passes the same z values to both
    candidate VAEs.  That prevents different random draws from masquerading as
    a model-quality difference.
    """

    outputs = []
    for start in range(0, len(latent_samples), batch_size):
        z = tf.convert_to_tensor(
            latent_samples[start:start + batch_size],
            dtype=tf.float32,
        )
        generated = vae.decoder.decoder_net(z, training=False)
        outputs.append(np.asarray(generated))
    return np.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# Mapping images into the frozen reference-VAE feature space
# ---------------------------------------------------------------------------

def extract_features(
    feature_model,
    images: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """
    Convert images into vectors in the frozen reference feature space.

    Some hidden layers may be spatial tensors rather than already-flat vectors,
    so each output is flattened to shape [batch, feature_dimension].
    """

    result = []
    for start in range(0, len(images), batch_size):
        batch = tf.convert_to_tensor(images[start:start + batch_size])
        features = feature_model(batch, training=False)
        features = tf.reshape(features, [tf.shape(features)[0], -1])
        result.append(np.asarray(features, dtype=np.float64))
    return np.concatenate(result, axis=0)


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

def feature_distribution_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the two sufficient statistics used by the Fréchet calculation.

    FID does not need the original feature vectors once their sample mean and
    covariance have been calculated.  In this module the raw-image feature set
    is fixed for the entire Pareto scan, so its (mean, covariance) pair can be
    calculated once in main() and reused for every Pareto point.

    At each Pareto point there are only two new distributions -- generated and
    reconstructed -- so each of those statistics is calculated once and reused
    in both pairwise comparisons that need it.  This removes redundant
    covariance calculations without changing the FID definition or result.
    """

    return (
        np.mean(features, axis=0),
        np.cov(features, rowvar=False),
    )


def frechet_distance_from_stats(stats_a, stats_b) -> float:
    """Fréchet distance from already-computed (mean, covariance) pairs."""

    mu_a, cov_a = stats_a
    mu_b, cov_b = stats_b
    diff = mu_a - mu_b

    eig_a, vec_a = np.linalg.eigh(cov_a)
    eig_a = np.clip(eig_a, 0.0, None)
    sqrt_cov_a = (vec_a * np.sqrt(eig_a)) @ vec_a.T

    middle = sqrt_cov_a @ cov_b @ sqrt_cov_a
    middle = 0.5 * (middle + middle.T)
    eig_middle = np.linalg.eigvalsh(middle)
    trace_sqrt = np.sum(np.sqrt(np.clip(eig_middle, 0.0, None)))

    fid = (
        diff @ diff
        + np.trace(cov_a)
        + np.trace(cov_b)
        - 2.0 * trace_sqrt
    )
    return float(max(fid, 0.0))


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Compatibility wrapper for callers that do not already have statistics."""

    return frechet_distance_from_stats(
        feature_distribution_stats(features_a),
        feature_distribution_stats(features_b),
    )


def polynomial_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Standard degree-3 polynomial kernel used by KID.

        k(x, y) = (x^T y / d + 1)^3

    where d is the feature-vector dimension.
    """

    d = x.shape[1]
    return (x @ y.T / float(d) + 1.0) ** 3


def unbiased_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    """
    Unbiased estimate of squared Maximum Mean Discrepancy (MMD^2).

    This is the core KID statistic.  Diagonal self-similarities are excluded
    from the within-set terms; that exclusion is what gives the usual unbiased
    finite-sample estimator.
    """

    m = len(x)
    n = len(y)
    if m < 2 or n < 2:
        raise ValueError("KID requires at least two samples from each set")

    # Pairwise similarities:
    #   k_xx -> within distribution X
    #   k_yy -> within distribution Y
    #   k_xy -> cross-distribution similarity
    k_xx = polynomial_kernel(x, x)
    k_yy = polynomial_kernel(y, y)
    k_xy = polynomial_kernel(x, y)

    term_xx = (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1))
    term_yy = (np.sum(k_yy) - np.trace(k_yy)) / (n * (n - 1))
    term_xy = np.mean(k_xy)

    return float(term_xx + term_yy - 2.0 * term_xy)


def kid(
    features_a: np.ndarray,
    features_b: np.ndarray,
    *,
    subset_size: int,
    subsets: int,
    seed: int,
) -> dict:
    """
    Estimate KID repeatedly on random fixed-size subsets.

    Repeated subsets are commonly used because:
        * KID is a finite-sample statistic;
        * subset estimates give a useful empirical spread;
        * pairwise kernel matrices scale quadratically with subset_size.

    The returned std is the spread across subset estimates, not uncertainty from
    repeated model training.
    """

    # If the caller asks for a subset larger than either feature set, use the
    # largest legal common subset rather than failing.
    subset_size = min(subset_size, len(features_a), len(features_b))
    if subset_size < 2:
        raise ValueError("KID subset_size must be at least 2")

    # Fixed RNG seed makes repeated evaluation of the same saved VAEs exactly
    # reproducible with respect to KID subset selection.
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(subsets):
        idx_a = rng.choice(len(features_a), subset_size, replace=False)
        idx_b = rng.choice(len(features_b), subset_size, replace=False)
        values.append(
            unbiased_mmd2(features_a[idx_a], features_b[idx_b])
        )

    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "subset_size": int(subset_size),
        "subsets": int(subsets),
    }


def calculate_pair_metrics(
    features_a,
    features_b,
    *,
    kid_subset_size,
    kid_subsets,
    kid_seed,
):
    """Calculate both distribution metrics from one pair of feature arrays."""

    return {
        "vae_feature_fid": frechet_distance(features_a, features_b),
        "vae_feature_kid": kid(
            features_a,
            features_b,
            subset_size=kid_subset_size,
            subsets=kid_subsets,
            seed=kid_seed,
        ),
    }


def gpu_polynomial_gram(features_a: np.ndarray, features_b: np.ndarray):
    """
    Build one degree-3 polynomial Gram matrix with TensorFlow float32 matmul.

    TensorFlow will place the large matrix multiplication on the GPU when a GPU
    is available.  This is the expensive O(N^2 d) part of KID and is exactly
    the kind of dense matrix operation the GPU handles efficiently.

    The feature extractor currently returns float64 arrays because the original
    NumPy implementation used that dtype throughout.  For the Gram matrices we
    deliberately cast to float32.  KID is a statistical comparison whose useful
    precision is dominated by finite-sample variation, not 64-bit arithmetic;
    float32 therefore provides ample numerical precision here while halving
    matrix memory and making GPU matmul substantially cheaper.

    The individual subset KID values are converted back to ordinary Python
    floats and finally accumulated in a NumPy float64 array when mean/std are
    reported.  Thus float32 is used only where it buys us the most: the large
    kernel-matrix computation.
    """

    a = tf.convert_to_tensor(features_a, dtype=tf.float32)
    b = tf.convert_to_tensor(features_b, dtype=tf.float32)
    d = tf.cast(tf.shape(a)[1], tf.float32)
    return tf.pow(tf.matmul(a, b, transpose_b=True) / d + 1.0, 3)


def make_kid_subset_plan(
    *,
    n_raw: int,
    n_generated: int,
    n_reconstructed: int,
    subset_size: int,
    subsets: int,
    seed: int,
):
    """
    Generate one seeded subset plan reused across all three KID comparisons.

    Each distribution receives its own independently sampled indices.  We do
    NOT force raw[i], generated[i], and reconstructed[i] to be treated as paired
    observations; KID is a distribution metric and does not require such a
    pairing.

    However, once a subset has been chosen for a distribution, those exact
    indices are reused everywhere that distribution appears.  For example, the
    same generated subset is used for generated-vs-raw and
    generated-vs-reconstructed.  This removes needless resampling work and also
    makes the three reported KID comparisons slightly easier to compare because
    they share the same underlying subset draws for common distributions.

    The seed makes the whole subset plan reproducible across reruns.
    """

    subset_size = min(
        subset_size,
        n_raw,
        n_generated,
        n_reconstructed,
    )
    if subset_size < 2:
        raise ValueError("KID subset_size must be at least 2")

    rng = np.random.default_rng(seed)
    plan = []
    for _ in range(subsets):
        plan.append({
            "raw": rng.choice(n_raw, subset_size, replace=False),
            "generated": rng.choice(n_generated, subset_size, replace=False),
            "reconstructed": rng.choice(
                n_reconstructed,
                subset_size,
                replace=False,
            ),
        })
    return plan, subset_size


def kid_from_precomputed_grams(
    self_gram_a,
    self_gram_b,
    cross_gram,
    subset_plan,
    *,
    index_name_a: str,
    index_name_b: str,
    subset_size: int,
) -> dict:
    """
    Calculate KID from precomputed Gram matrices.

    No feature-space matmul occurs inside the subset loop.  The expensive
    polynomial kernel has already been evaluated for every pair of images in
    the full distributions.  Each subset iteration therefore does only:

        1. gather the relevant rows/columns from those precomputed matrices;
        2. remove diagonal self-similarities for the unbiased within-set terms;
        3. reduce the three matrices to the scalar MMD^2/KID estimate.

    This is mathematically the same subset estimator as before; the difference
    is that repeated matrix multiplication has been replaced by cheap indexing
    and reduction.
    """

    values = []

    for indices in subset_plan:
        idx_a = tf.convert_to_tensor(indices[index_name_a], dtype=tf.int32)
        idx_b = tf.convert_to_tensor(indices[index_name_b], dtype=tf.int32)

        k_aa = tf.gather(
            tf.gather(self_gram_a, idx_a, axis=0),
            idx_a,
            axis=1,
        )
        k_bb = tf.gather(
            tf.gather(self_gram_b, idx_b, axis=0),
            idx_b,
            axis=1,
        )
        k_ab = tf.gather(
            tf.gather(cross_gram, idx_a, axis=0),
            idx_b,
            axis=1,
        )

        m = tf.cast(subset_size, tf.float32)
        denom = m * (m - 1.0)

        term_aa = (
            tf.reduce_sum(k_aa) - tf.linalg.trace(k_aa)
        ) / denom
        term_bb = (
            tf.reduce_sum(k_bb) - tf.linalg.trace(k_bb)
        ) / denom
        term_ab = tf.reduce_mean(k_ab)

        values.append(float((term_aa + term_bb - 2.0 * term_ab).numpy()))

    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "subset_size": int(subset_size),
        "subsets": int(len(values)),
    }


def calculate_triplet_metrics(
    *,
    raw_features: np.ndarray,
    generated_features: np.ndarray,
    reconstructed_features: np.ndarray,
    raw_stats,
    kid_subset_size: int,
    kid_subsets: int,
    kid_seed: int,
):
    """
    Calculate all three FID/KID comparisons while reusing expensive work.

    FID:
        * raw mean/covariance is supplied by the caller and reused globally;
        * generated/reconstructed mean/covariance are computed once each.

    KID:
        * six full Gram matrices are computed once per Pareto point;
        * all large Gram matmuls use TensorFlow float32 (GPU when available);
        * subset indices are generated once and reused across comparisons.

    Why exactly six Gram matrices?
    --------------------------------
    There are three distributions at a Pareto point:

        R = raw images
        G = generated images
        C = reconstructed images

    The unbiased KID estimator for any pair needs two within-distribution
    kernels and one cross-distribution kernel.  Across all three comparisons we
    can share those matrices, so the complete unique set is:

        R-R, G-G, C-C      three self-kernels
        G-R, C-R, G-C      three cross-kernels

    Once these six matrices exist, all 20 (or however many) KID subsets for all
    three comparisons can be calculated without another feature-space matmul.
    Under the old implementation the same self-kernels and cross-kernels were
    rebuilt repeatedly inside every subset call.
    """

    generated_stats = feature_distribution_stats(generated_features)
    reconstructed_stats = feature_distribution_stats(reconstructed_features)

    subset_plan, subset_size = make_kid_subset_plan(
        n_raw=len(raw_features),
        n_generated=len(generated_features),
        n_reconstructed=len(reconstructed_features),
        subset_size=kid_subset_size,
        subsets=kid_subsets,
        seed=kid_seed,
    )

    # Precompute the complete kernel information for this Pareto point.
    #
    # With three distributions, there are only six unique matrices we ever
    # need: three self-kernels (R-R, G-G, C-C) and the three pairwise
    # cross-kernels (G-R, C-R, G-C).  Reusing these is the major KID speedup.
    #
    # At the default 2000 images, one 2000x2000 float32 Gram matrix is about
    # 16 MB, so all six occupy roughly 96 MB -- inexpensive relative to the
    # available GPU memory on the intended workstation.
    gram_raw_raw = gpu_polynomial_gram(raw_features, raw_features)
    gram_gen_gen = gpu_polynomial_gram(generated_features, generated_features)
    gram_recon_recon = gpu_polynomial_gram(
        reconstructed_features,
        reconstructed_features,
    )
    gram_gen_raw = gpu_polynomial_gram(generated_features, raw_features)
    gram_recon_raw = gpu_polynomial_gram(
        reconstructed_features,
        raw_features,
    )
    gram_gen_recon = gpu_polynomial_gram(
        generated_features,
        reconstructed_features,
    )

    generated_vs_raw_kid = kid_from_precomputed_grams(
        gram_gen_gen,
        gram_raw_raw,
        gram_gen_raw,
        subset_plan,
        index_name_a="generated",
        index_name_b="raw",
        subset_size=subset_size,
    )
    reconstructed_vs_raw_kid = kid_from_precomputed_grams(
        gram_recon_recon,
        gram_raw_raw,
        gram_recon_raw,
        subset_plan,
        index_name_a="reconstructed",
        index_name_b="raw",
        subset_size=subset_size,
    )
    generated_vs_reconstructed_kid = kid_from_precomputed_grams(
        gram_gen_gen,
        gram_recon_recon,
        gram_gen_recon,
        subset_plan,
        index_name_a="generated",
        index_name_b="reconstructed",
        subset_size=subset_size,
    )

    return {
        "generated_vs_raw": {
            "vae_feature_fid": frechet_distance_from_stats(
                generated_stats,
                raw_stats,
            ),
            "vae_feature_kid": generated_vs_raw_kid,
        },
        "reconstructed_vs_raw": {
            "vae_feature_fid": frechet_distance_from_stats(
                reconstructed_stats,
                raw_stats,
            ),
            "vae_feature_kid": reconstructed_vs_raw_kid,
        },
        "generated_vs_reconstructed": {
            "vae_feature_fid": frechet_distance_from_stats(
                generated_stats,
                reconstructed_stats,
            ),
            "vae_feature_kid": generated_vs_reconstructed_kid,
        },
    }


# ---------------------------------------------------------------------------
# Packaging results
# ---------------------------------------------------------------------------

def checkpoint_summary(record: dict) -> dict:
    return {
        "epoch": int(record["epoch"]),
        "step": int(record["step"]),
        "iteration": int(record["iteration"]),
        "recon_loss": float(record["recon_loss"]),
        "kl_loss": float(record["kl_loss"]),
        "checkpoint": str(record["checkpoint_path"]),
    }


def evaluate_candidate(
    vae: VAE_Model,
    raw_images: np.ndarray,
    latent_samples: np.ndarray,
    feature_model,
    *,
    batch_size: int,
    raw_features: np.ndarray,
    raw_stats,
    kid_subset_size: int,
    kid_subsets: int,
    kid_seed: int,
):
    """
    Produce all image sets for one candidate VAE and evaluate three comparisons.

    raw_features are supplied by the caller because the raw image set and
    reference encoder are common to BOTH experiments; computing them once avoids
    duplicated work and guarantees exact consistency.

    raw_stats is the cached (mean, covariance) pair for those same raw features.
    It is reused by every Pareto point, so neither raw feature extraction nor
    raw covariance calculation is repeated during the frontier scan.

    After generated/reconstructed features are extracted, calculate_triplet_metrics()
    handles all three comparisons together.  Doing them as one triplet is what
    allows both FID statistics and KID Gram matrices to be shared instead of
    recomputed independently for each comparison.
    """

    # Candidate-specific image sets.
    recon_images = reconstruct_mu(vae, raw_images, batch_size)
    generated_images = generate_images(vae, latent_samples, batch_size)

    # Map both sets into the same frozen reference-VAE feature space.
    recon_features = extract_features(feature_model, recon_images, batch_size)
    generated_features = extract_features(feature_model, generated_images, batch_size)

    return calculate_triplet_metrics(
        raw_features=raw_features,
        generated_features=generated_features,
        reconstructed_features=recon_features,
        raw_stats=raw_stats,
        kid_subset_size=kid_subset_size,
        kid_subsets=kid_subsets,
        kid_seed=kid_seed,
    )




def latest_checkpoint_at_or_before(checkpoints, iteration: int) -> dict:
    """Return the chronologically latest checkpoint not after ``iteration``."""

    eligible = [
        record
        for record in checkpoints
        if int(record["iteration"]) <= int(iteration)
    ]
    if not eligible:
        raise ValueError(
            f"No periodic checkpoint exists at or before iteration {iteration}"
        )
    return max(eligible, key=lambda record: int(record["iteration"]))


def pareto_record_key(record: dict) -> tuple[int, int, int]:
    """Stable key for matching one recorded Pareto point to saved results."""

    return (
        int(record["epoch"]),
        int(record["step"]),
        int(record["iteration"]),
    )


def load_pareto_progress(progress_path: Path | None) -> dict:
    """
    Load already-completed Pareto results from a prior interrupted run.

    This is primarily fault tolerance, not a performance optimization during a
    successful first run.  The expensive part of one Pareto evaluation is the
    image generation/feature extraction/FID/KID work; persisting its finished
    result means that work does not have to be repeated after a crash or manual
    interruption.

    Training replay is different: if a later unfinished Pareto point lies in
    the same checkpoint interval, the model may still need to replay through
    earlier training steps to reach it.  Resume therefore saves completed
    metric evaluations, but cannot always eliminate all intervening training.
    """

    if progress_path is None or not progress_path.is_file():
        return {}

    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    completed = {}
    for item in payload.get("pareto_points", []):
        if not all(name in item for name in ("epoch", "step", "iteration", "metrics")):
            continue
        completed[
            (int(item["epoch"]), int(item["step"]), int(item["iteration"]))
        ] = item
    return completed


def save_pareto_progress(
    progress_path: Path | None,
    *,
    expt_name: str,
    results_by_key: dict,
):
    """
    Atomically persist completed Pareto metrics after each evaluated point.

    The temporary-file + replace pattern avoids leaving a truncated JSON file
    if the process is interrupted during the write.  The temporary file is
    written completely first and only then atomically replaces the progress
    file, so an interruption should leave either the previous valid progress
    file or the new complete one rather than half-written JSON.

    This intentionally writes after every newly evaluated Pareto point.  The
    JSON I/O is tiny compared with generating images and computing distribution
    metrics, so the small write overhead buys useful insurance for a long scan.
    """

    if progress_path is None:
        return

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        results_by_key.values(),
        key=lambda item: int(item["iteration"]),
    )
    payload = {
        "experiment": expt_name,
        "status": "in_progress",
        "pareto_points": ordered,
    }

    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(progress_path)


def evaluate_all_pareto_points(
    *,
    expt_dir: Path,
    frontier: list[dict],
    checkpoints: list[dict],
    raw_images: np.ndarray,
    latent_samples: np.ndarray,
    feature_model,
    batch_size: int,
    raw_features: np.ndarray,
    raw_stats,
    kid_subset_size: int,
    kid_subsets: int,
    kid_seed: int,
    save_weights_for_keys=None,
    progress_path: Path | None = None,
):
    """
    Rederive and evaluate every retained Pareto point in one experiment.

    Pareto points are grouped by the latest periodic checkpoint at or before
    them.  Each checkpoint is loaded once, and training is replayed forward
    through all Pareto points belonging to that checkpoint interval.  Thus a
    50-epoch checkpoint cadence causes at most roughly 50 epochs of replay per
    checkpoint interval instead of independently replaying from a checkpoint
    for every Pareto point.

    ``save_weights_for_keys`` optionally maps a Pareto record key to a path.
    This is used by the pairwise intersection analysis to preserve the two
    rederived intersection-side VAEs without saving a VAE for every Pareto
    point.
    """

    save_weights_for_keys = save_weights_for_keys or {}
    frontier = sorted(frontier, key=lambda record: int(record["iteration"]))
    if not frontier:
        return []

    cfg = TrainerConfig.from_file(expt_dir / "config.ini")
    all_loss_records = read_loss_records(expt_dir)
    records_by_key = {
        (int(record["epoch"]), int(record["step"])): record
        for record in all_loss_records
    }

    # Determine the final recorded step in every epoch.  This lets replay
    # begin in the following epoch when a periodic checkpoint was saved at the
    # end of an epoch, as is normal for this project.
    max_step_by_epoch = {}
    for record in all_loss_records:
        epoch = int(record["epoch"])
        step = int(record["step"])
        max_step_by_epoch[epoch] = max(max_step_by_epoch.get(epoch, -1), step)

    # Assign each Pareto record to the most recent recoverable checkpoint.
    groups = {}
    for record in frontier:
        checkpoint = latest_checkpoint_at_or_before(
            checkpoints,
            int(record["iteration"]),
        )
        checkpoint_path = str(checkpoint["checkpoint_path"])
        if checkpoint_path not in groups:
            groups[checkpoint_path] = {
                "checkpoint": checkpoint,
                "targets": [],
            }
        groups[checkpoint_path]["targets"].append(record)

    ordered_groups = sorted(
        groups.values(),
        key=lambda group: int(group["checkpoint"]["iteration"]),
    )

    training_dataset = make_candidate_training_dataset(
        cfg,
        scratch_dir=expt_dir / "_fid_kid_pareto_rederive",
    )

    results_by_key = load_pareto_progress(progress_path)
    if results_by_key:
        print(
            f"{expt_dir.name}: resuming with "
            f"{len(results_by_key)} completed Pareto point(s)"
        )

    for group_index, group in enumerate(ordered_groups, start=1):
        checkpoint = group["checkpoint"]
        # Completed Pareto points do not need image generation or FID/KID a
        # second time.  They are removed from this group's evaluation targets.
        # If unfinished points remain later in the same checkpoint interval,
        # training still starts from the checkpoint and replays forward as
        # necessary to reconstruct those later model states.
        targets = sorted(
            [
                record
                for record in group["targets"]
                if pareto_record_key(record) not in results_by_key
            ],
            key=lambda record: int(record["iteration"]),
        )
        if not targets:
            continue

        target_by_epoch_step = {
            (int(record["epoch"]), int(record["step"])): record
            for record in targets
        }

        print()
        group_start = datetime.now()
        print(
            f"{expt_dir.name}: Pareto checkpoint group "
            f"{group_index}/{len(ordered_groups)} "
            f"started {group_start:%Y-%m-%d %H:%M:%S}"
        )
        print(
            f"  loading epoch {checkpoint['epoch']} step {checkpoint['step']} "
            f"for {len(targets)} Pareto point(s)"
        )

        vae, _ = load_experiment_checkpoint(expt_dir, checkpoint)
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Build this optimizer's slot variables before tracing. Each
        # checkpoint group gets its own optimizer and its own tf.function,
        # avoiding TensorFlow's "singleton tf.Variable" error when multiple
        # optimizers are used with one globally traced function.
        optimizer.build(vae.vae_net.trainable_weights)

        @tf.function(reduce_retracing=True)
        def compiled_train_step(x_batch_train, kl_weight_tensor):
            return rederivation_train_step(
                x_batch_train,
                kl_weight_tensor,
                vae,
                loss_fn,
                optimizer,
            )

        cp_epoch = int(checkpoint["epoch"])
        cp_step = int(checkpoint["step"])
        cp_iteration = int(checkpoint["iteration"])
        steps_since_checkpoint = 0

        def evaluate_target(record, final_recon, final_kl):
            key = pareto_record_key(record)

            # Resume protection: FID/KID evaluation is substantially more
            # expensive than checking the saved-results dictionary.  The
            # checkpoint-group setup already filters completed Pareto points,
            # but keep this second check immediately before metric calculation
            # so a previously saved result can never be recalculated
            # accidentally.
            if key in results_by_key:
                print(
                    f"  Pareto epoch {record['epoch']} step {record['step']}: "
                    "FID/KID already saved; skipping metric calculation"
                )
                return

            metrics = evaluate_candidate(
                vae,
                raw_images,
                latent_samples,
                feature_model,
                batch_size=batch_size,
                raw_features=raw_features,
                raw_stats=raw_stats,
                kid_subset_size=kid_subset_size,
                kid_subsets=kid_subsets,
                kid_seed=kid_seed,
            )

            save_path = save_weights_for_keys.get(key)
            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                vae.vae_net.save_weights(save_path, overwrite=True)

            result = {
                "epoch": int(record["epoch"]),
                "step": int(record["step"]),
                "iteration": int(record["iteration"]),
                "recon_loss": float(record["recon_loss"]),
                "kl_loss": float(record["kl_loss"]),
                "starting_checkpoint": checkpoint_summary(checkpoint),
                "steps_rederived": int(steps_since_checkpoint),
                "rederived_training_step_recon_loss": float(final_recon),
                "rederived_training_step_kl_loss": float(final_kl),
                "metrics": metrics,
            }
            if save_path is not None:
                result["saved_rederived_weights"] = str(save_path)

            results_by_key[key] = result
            save_pareto_progress(
                progress_path,
                expt_name=expt_dir.name,
                results_by_key=results_by_key,
            )

            generated = metrics["generated_vs_raw"]
            print(
                f"  Pareto epoch {record['epoch']} step {record['step']}: "
                f"FID={generated['vae_feature_fid']:.6g}, "
                f"KID={generated['vae_feature_kid']['mean']:.6g}"
            )

        # A target may coincide exactly with the periodic checkpoint itself.
        checkpoint_key = (cp_epoch, cp_step)
        if checkpoint_key in target_by_epoch_step:
            record = target_by_epoch_step[checkpoint_key]
            evaluate_target(
                record,
                float(checkpoint["recon_loss"]),
                float(checkpoint["kl_loss"]),
            )

        final_target = targets[-1]
        final_epoch = int(final_target["epoch"])
        final_step = int(final_target["step"])

        if cp_step >= max_step_by_epoch.get(cp_epoch, cp_step):
            first_epoch = cp_epoch + 1
        else:
            first_epoch = cp_epoch

        for epoch in range(first_epoch, final_epoch + 1):
            # Explicit progress replaces TensorFlow's noisy end-of-dataset
            # OUT_OF_RANGE messages as the visible indication that replay is
            # advancing normally through epochs.
            if (
                epoch == first_epoch
                or epoch == final_epoch
                or (epoch - first_epoch) % 10 == 0
            ):
                print(f"  replaying epoch {epoch} / {final_epoch}")

            for step, x_batch_train in enumerate(training_dataset):
                if epoch == cp_epoch and step <= cp_step:
                    continue
                if epoch == final_epoch and step > final_step:
                    break

                original = records_by_key.get((epoch, step))
                if original is None:
                    raise ValueError(
                        f"{expt_dir.name}: no loss record for epoch {epoch}, "
                        f"step {step}; cannot recover original KL-weight schedule"
                    )

                kl_weight_tensor = tf.convert_to_tensor(
                    float(original["kl_weight"]),
                    dtype=tf.float32,
                )
                loss_recon, loss_kl = compiled_train_step(
                    x_batch_train,
                    kl_weight_tensor,
                )
                steps_since_checkpoint += 1

                target_record = target_by_epoch_step.get((epoch, step))
                if target_record is not None:
                    evaluate_target(
                        target_record,
                        float(loss_recon.numpy()),
                        float(loss_kl.numpy()),
                    )

                if epoch == final_epoch and step == final_step:
                    break

    # Preserve the Pareto frontier's chronological ordering in output.
    missing = [
        pareto_record_key(record)
        for record in frontier
        if pareto_record_key(record) not in results_by_key
    ]
    if missing:
        raise RuntimeError(
            f"{expt_dir.name}: failed to evaluate {len(missing)} Pareto point(s)"
        )

    return [results_by_key[pareto_record_key(record)] for record in frontier]


def write_pareto_results_text(path: Path, expt_name: str, pareto_results: list[dict]):
    """Save one compact human-readable table containing every Pareto metric."""

    lines = []
    lines.append(f"{expt_name} - VAE-feature FID/KID at Pareto Points")
    lines.append("=" * (len(lines[0])))
    lines.append("")
    lines.append(
        "Lower FID/KID is better. Values use the fixed reference-VAE feature "
        "space, not standard Inception features."
    )
    lines.append("")
    lines.append(
        "Epoch  Step  Iteration     Recon          KL      "
        "Gen/Raw FID   Gen/Raw KID    Recon/Raw FID  Recon/Raw KID"
    )
    lines.append("-" * 112)

    for result in pareto_results:
        gen_raw = result["metrics"]["generated_vs_raw"]
        recon_raw = result["metrics"]["reconstructed_vs_raw"]
        lines.append(
            f"{result['epoch']:5d}  "
            f"{result['step']:4d}  "
            f"{result['iteration']:9d}  "
            f"{result['recon_loss']:10.6g}  "
            f"{result['kl_loss']:10.6g}  "
            f"{gen_raw['vae_feature_fid']:11.6g}  "
            f"{gen_raw['vae_feature_kid']['mean']:12.6g}  "
            f"{recon_raw['vae_feature_fid']:13.6g}  "
            f"{recon_raw['vae_feature_kid']['mean']:13.6g}"
        )

    lines.append("")
    lines.append("Detailed generated-vs-reconstructed metrics and KID standard")
    lines.append("deviations are retained in the companion JSON file.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_human_readable_report(
    path: Path,
    *,
    args,
    output: dict,
    expt_a_name: str,
    expt_b_name: str,
):
    """Write the main result and provenance in a compact plain-text report."""

    lines = []
    lines.append("VAE Pareto Intersection FID/KID Comparison")
    lines.append("=" * 44)
    lines.append("")
    lines.append(
        f"Experiments: {expt_a_name} vs {expt_b_name}"
    )
    lines.append(
        f"Reference VAE: ref_vae/expt_{args.ref_expt}"
    )
    lines.append(
        f"Reference feature layer: {args.feature_layer}"
    )
    lines.append(
        f"Images per distribution: {args.n_images}"
    )
    lines.append("")
    lines.append("IMPORTANT")
    lines.append("---------")
    lines.append(
        "The two intersection VAEs were rederived by continuing training "
        "from the nearest earlier periodic checkpoint."
    )
    lines.append(
        "Only model weights were checkpointed. Optimizer state, RNG state, "
        "tf.data iterator state, and adaptive-controller state were not."
    )
    lines.append(
        "Therefore these are approximate rederivations at the same nominal "
        "epoch/step, not bit-for-bit recoveries of the historical models."
    )
    lines.append("")
    lines.append("Pareto target")
    lines.append("-------------")
    intersection = output["intersection"]
    lines.append(f"Type: {intersection['kind']}")
    lines.append(
        f"Target reconstruction loss: "
        f"{intersection['target_recon_loss']:.8g}"
    )
    lines.append(
        f"Target KL loss: {intersection['target_kl_loss']:.8g}"
    )
    lines.append("")

    for expt_name in (expt_a_name, expt_b_name):
        item = output[expt_name]
        frontier = item["intersection_frontier_record"]
        checkpoint = item["checkpoint"]
        rederived = item["rederived_state"]

        lines.append(expt_name)
        lines.append("-" * len(expt_name))
        lines.append(
            "Intersection-side original record: "
            f"epoch {frontier['epoch']}, step {frontier['step']}, "
            f"iteration {frontier['iteration']}"
        )
        lines.append(
            f"  original recon={frontier['recon_loss']:.8g}, "
            f"KL={frontier['kl_loss']:.8g}"
        )
        lines.append(
            "Starting checkpoint: "
            f"epoch {checkpoint['epoch']}, step {checkpoint['step']}, "
            f"iteration {checkpoint['iteration']}"
        )
        lines.append(
            f"  checkpoint recon={checkpoint['recon_loss']:.8g}, "
            f"KL={checkpoint['kl_loss']:.8g}"
        )
        lines.append(
            f"  checkpoint file: {checkpoint['checkpoint']}"
        )
        lines.append(
            f"Gradient steps rederived: {rederived['steps_rederived']}"
        )
        lines.append(
            "Rederived final training-step losses: "
            f"recon={rederived['final_rederived_recon_loss']:.8g}, "
            f"KL={rederived['final_rederived_kl_loss']:.8g}"
        )
        lines.append(
            f"Saved rederived weights: {item['rederived_weights']}"
        )
        lines.append("")
        lines.append("Image-distribution metrics:")
        for comparison, values in item["metrics"].items():
            kid_values = values["vae_feature_kid"]
            lines.append(
                f"  {comparison}:"
            )
            lines.append(
                f"    VAE-feature FID: {values['vae_feature_fid']:.8g}"
            )
            lines.append(
                f"    VAE-feature KID: {kid_values['mean']:.8g} "
                f"+/- {kid_values['std']:.8g}"
            )
            lines.append(
                f"    KID subsets: {kid_values['subsets']} x "
                f"{kid_values['subset_size']} images"
            )
        lines.append("")

    lines.append("Interpretation")
    lines.append("--------------")
    lines.append(
        "Lower VAE-feature FID and KID indicate closer distributions in the "
        "fixed reference-VAE feature space."
    )
    lines.append(
        "generated_vs_raw is the primary generation-quality comparison; "
        "reconstructed_vs_raw is a reconstruction baseline; "
        "generated_vs_reconstructed is diagnostic."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Command-line interface and end-to-end workflow
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recover the nearest earlier periodic checkpoints, rederive the VAEs "
            "through the Pareto intersection-side epoch/step, and calculate "
            "VAE-feature FID/KID."
        )
    )
    parser.add_argument(
        "--expts",
        type=int,
        nargs=2,
        required=True,
        metavar=("EXPT_A", "EXPT_B"),
    )
    parser.add_argument(
        "--ref_expt",
        type=int,
        required=True,
        help="Reference VAE experiment number under ref_vae/expt_N",
    )
    parser.add_argument(
        "--parent_dir",
        type=Path,
        default=Path("expts"),
    )
    parser.add_argument(
        "--ref_root",
        type=Path,
        default=Path("ref_vae"),
    )
    parser.add_argument(
        "--skip_fraction",
        type=float,
        default=0.10,
        help="Chronological fraction excluded before Pareto calculation",
    )
    parser.add_argument(
        "--feature_layer",
        default="lrelu_4",
        help=(
            "Reference encoder layer used as image features "
            "(default: lrelu_4, the 1024-D hidden layer before latent_mu)"
        ),
    )
    parser.add_argument("--n_images", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--kid_subsets", type=int, default=20)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=12345,
        help="Seed for common prior samples and KID subsets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path; a human-readable .txt report is also written",
    )
    return parser.parse_args()


def main():
    run_start = perf_counter()
    args = parse_args()

    if not 0.0 <= args.skip_fraction < 1.0:
        raise ValueError("--skip_fraction must satisfy 0 <= value < 1")
    if args.n_images < 2:
        raise ValueError("--n_images must be at least 2")

    # ------------------------------------------------------------------
    # Phase 1: locate the comparison target in the COMPLETE loss histories.
    # ------------------------------------------------------------------
    expt_a = resolve_experiment_dir(args.parent_dir, args.expts[0])
    expt_b = resolve_experiment_dir(args.parent_dir, args.expts[1])

    frontier_a = retained_pareto_records(expt_a, args.skip_fraction)
    frontier_b = retained_pareto_records(expt_b, args.skip_fraction)

    # Periodic checkpoints are sparse in time, so they are only a set of
    # recoverable approximations to the exact Pareto target.
    checkpoints_a = find_checkpoint_records(expt_a)
    checkpoints_b = find_checkpoint_records(expt_b)

    target = choose_target(
        frontier_a,
        frontier_b,
        checkpoints_a,
        checkpoints_b,
    )

    cp_a = target["checkpoint_a"]
    cp_b = target["checkpoint_b"]

    print()
    print("Pareto comparison target")
    print("========================")
    print(f"type:       {target['kind']}")
    print(f"recon_loss: {target['target_recon_loss']:.8g}")
    print(f"kl_loss:    {target['target_kl_loss']:.8g}")
    print()
    frontier_record_a = target["frontier_record_a"]
    frontier_record_b = target["frontier_record_b"]

    print(f"{expt_a.name}: intersection-side frontier record")
    print(
        f"  epoch {frontier_record_a['epoch']} "
        f"step {frontier_record_a['step']} "
        f"iteration {frontier_record_a['iteration']}"
    )
    print(f"{expt_a.name}: nearest EARLIER checkpoint")
    print(f"  epoch {cp_a['epoch']} step {cp_a['step']}")
    print(f"  recon={cp_a['recon_loss']:.8g} kl={cp_a['kl_loss']:.8g}")
    print(f"  {cp_a['checkpoint_path']}")
    print()

    print(f"{expt_b.name}: intersection-side frontier record")
    print(
        f"  epoch {frontier_record_b['epoch']} "
        f"step {frontier_record_b['step']} "
        f"iteration {frontier_record_b['iteration']}"
    )
    print(f"{expt_b.name}: nearest EARLIER checkpoint")
    print(f"  epoch {cp_b['epoch']} step {cp_b['step']}")
    print(f"  recon={cp_b['recon_loss']:.8g} kl={cp_b['kl_loss']:.8g}")
    print(f"  {cp_b['checkpoint_path']}")
    print()

    # ------------------------------------------------------------------
    # Phase 2: resolve output paths and load candidate configs.
    # ------------------------------------------------------------------
    # Pairwise intersection reports are comparison artifacts, so keep them
    # together under pareto_comps/Intersections.  --output may override only
    # the filename; the comparison directory remains fixed.
    intersection_dir = Path("pareto_comps") / "Intersections"
    intersection_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        output_name = (
            f"VAE_FID_KID_expt_{args.expts[0]}_vs_{args.expts[1]}.json"
        )
    else:
        output_name = args.output.name

    output_path = intersection_dir / output_name
    report_path = output_path.with_suffix(".txt")

    # Experiment-specific FID/KID artifacts stay with the experiment that
    # produced them.  The rederived intersection weights are also model
    # artifacts for a specific experiment, so they live in the same tree.
    fid_kid_dir_a = expt_a / "FID_KID"
    fid_kid_dir_b = expt_b / "FID_KID"
    fid_kid_dir_a.mkdir(parents=True, exist_ok=True)
    fid_kid_dir_b.mkdir(parents=True, exist_ok=True)

    rederived_dir_a = fid_kid_dir_a / "rederived_intersections"
    rederived_dir_b = fid_kid_dir_b / "rederived_intersections"
    rederived_dir_a.mkdir(parents=True, exist_ok=True)
    rederived_dir_b.mkdir(parents=True, exist_ok=True)

    rederived_weights_a = (
        rederived_dir_a
        / f"{expt_a.name}_intersection_rederived.weights.h5"
    )
    rederived_weights_b = (
        rederived_dir_b
        / f"{expt_b.name}_intersection_rederived.weights.h5"
    )

    cfg_a = TrainerConfig.from_file(expt_a / "config.ini")
    cfg_b = TrainerConfig.from_file(expt_b / "config.ini")

    # ------------------------------------------------------------------
    # Phase 3: load the ONE frozen reference feature extractor.
    # ------------------------------------------------------------------
    ref_vae, ref_cfg, ref_dir = load_reference_vae(
        args.ref_root,
        args.ref_expt,
    )
    feature_model = make_reference_feature_model(
        ref_vae,
        args.feature_layer,
    )

    if cfg_a.image_size != ref_cfg.image_size or cfg_b.image_size != ref_cfg.image_size:
        raise ValueError("Candidate and reference VAEs must use the same image_size")
    if cfg_a.latent_dim != cfg_b.latent_dim:
        raise ValueError("The two candidate VAEs must use the same latent_dim")

    # ------------------------------------------------------------------
    # Phase 4: create one fixed real-image evaluation set.
    # ------------------------------------------------------------------
    validation_dataset = make_reference_validation_dataset(
        ref_cfg,
        scratch_dir=ref_dir / "_eval_dataset",
    )
    raw_images = collect_images(validation_dataset, args.n_images)
    raw_features = extract_features(
        feature_model,
        raw_images,
        args.batch_size,
    )
    # FID only needs mean/covariance once the raw features are known.  The raw
    # evaluation population never changes during this run, so cache those
    # statistics once here and reuse them for every Pareto point in both
    # experiments.
    raw_stats = feature_distribution_stats(raw_features)

    # ------------------------------------------------------------------
    # Phase 5: create one fixed prior sample set.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.eval_seed)
    latent_samples = rng.standard_normal(
        (args.n_images, cfg_a.latent_dim),
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Phase 6: evaluate EVERY retained Pareto point for both experiments.
    # ------------------------------------------------------------------
    # Checkpoints are recovery infrastructure only.  They are not themselves
    # the evaluation targets.  Each actual Pareto record is rederived and
    # evaluated.  The two intersection-side records are included in this same
    # pass and their rederived weights are preserved separately.
    intersection_key_a = pareto_record_key(frontier_record_a)
    intersection_key_b = pareto_record_key(frontier_record_b)

    pareto_json_a = fid_kid_dir_a / "Pareto_FID_KID.json"
    pareto_json_b = fid_kid_dir_b / "Pareto_FID_KID.json"
    pareto_text_a = fid_kid_dir_a / "Pareto_FID_KID.txt"
    pareto_text_b = fid_kid_dir_b / "Pareto_FID_KID.txt"

    pareto_results_a = evaluate_all_pareto_points(
        expt_dir=expt_a,
        frontier=frontier_a,
        checkpoints=checkpoints_a,
        raw_images=raw_images,
        latent_samples=latent_samples,
        feature_model=feature_model,
        batch_size=args.batch_size,
        raw_features=raw_features,
        raw_stats=raw_stats,
        kid_subset_size=args.kid_subset_size,
        kid_subsets=args.kid_subsets,
        kid_seed=args.eval_seed + 100,
        save_weights_for_keys={
            intersection_key_a: rederived_weights_a,
        },
        progress_path=pareto_json_a,
    )

    pareto_results_b = evaluate_all_pareto_points(
        expt_dir=expt_b,
        frontier=frontier_b,
        checkpoints=checkpoints_b,
        raw_images=raw_images,
        latent_samples=latent_samples,
        feature_model=feature_model,
        batch_size=args.batch_size,
        raw_features=raw_features,
        raw_stats=raw_stats,
        kid_subset_size=args.kid_subset_size,
        kid_subsets=args.kid_subsets,
        kid_seed=args.eval_seed + 100,
        save_weights_for_keys={
            intersection_key_b: rederived_weights_b,
        },
        progress_path=pareto_json_b,
    )

    pareto_by_key_a = {
        (item["epoch"], item["step"], item["iteration"]): item
        for item in pareto_results_a
    }
    pareto_by_key_b = {
        (item["epoch"], item["step"], item["iteration"]): item
        for item in pareto_results_b
    }
    intersection_result_a = pareto_by_key_a[intersection_key_a]
    intersection_result_b = pareto_by_key_b[intersection_key_b]

    # Save the complete Pareto tables separately so they remain useful even
    # without the pairwise intersection report.
    pareto_payload_common = {
        "metric_definition": {
            "feature_extractor": f"ref_vae/expt_{args.ref_expt}",
            "feature_layer": args.feature_layer,
            "note": (
                "These are VAE-feature Fréchet/KID values, not standard "
                "Inception FID/KID. Each row is an actual Pareto record; "
                "periodic checkpoints are used only to rederive that state."
            ),
            "n_images": args.n_images,
            "generated_latents": (
                "same fixed N(0,I) latent samples used for every Pareto point"
            ),
        },
    }

    with pareto_json_a.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                **pareto_payload_common,
                "experiment": expt_a.name,
                "status": "complete",
                "pareto_points": pareto_results_a,
            },
            fh,
            indent=2,
        )
    with pareto_json_b.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                **pareto_payload_common,
                "experiment": expt_b.name,
                "status": "complete",
                "pareto_points": pareto_results_b,
            },
            fh,
            indent=2,
        )

    write_pareto_results_text(pareto_text_a, expt_a.name, pareto_results_a)
    write_pareto_results_text(pareto_text_b, expt_b.name, pareto_results_b)

    # ------------------------------------------------------------------
    # Phase 7: write the pairwise intersection report using metrics already
    #          calculated for the two relevant Pareto records.
    # ------------------------------------------------------------------
    def intersection_rederived_state(item):
        return {
            "steps_rederived": int(item["steps_rederived"]),
            "final_rederived_recon_loss": float(
                item["rederived_training_step_recon_loss"]
            ),
            "final_rederived_kl_loss": float(
                item["rederived_training_step_kl_loss"]
            ),
        }

    output = {
        "metric_definition": {
            "feature_extractor": f"ref_vae/expt_{args.ref_expt}",
            "feature_layer": args.feature_layer,
            "note": (
                "These are VAE-feature Fréchet/KID values, not standard "
                "Inception FID/KID. Intersection metrics are references to "
                "the full Pareto-point evaluation."
            ),
            "n_images": args.n_images,
            "raw_image_set": (
                "first n_images from the common validation split reconstructed "
                "through the current Datasets logic"
            ),
            "generated_latents": (
                "same fixed N(0,I) latent samples used for both candidates"
            ),
        },
        "intersection": {
            "kind": target["kind"],
            "target_recon_loss": target["target_recon_loss"],
            "target_kl_loss": target["target_kl_loss"],
            "intersection_count": target["intersection_count"],
        },
        expt_a.name: {
            "intersection_frontier_record": {
                "epoch": int(frontier_record_a["epoch"]),
                "step": int(frontier_record_a["step"]),
                "iteration": int(frontier_record_a["iteration"]),
                "recon_loss": float(frontier_record_a["recon_loss"]),
                "kl_loss": float(frontier_record_a["kl_loss"]),
            },
            "checkpoint": intersection_result_a["starting_checkpoint"],
            "distance_to_intersection_target": target["distance_a"],
            "rederived_state": intersection_rederived_state(
                intersection_result_a
            ),
            "rederived_weights": str(rederived_weights_a),
            "metrics": intersection_result_a["metrics"],
            "all_pareto_results_json": str(pareto_json_a),
            "all_pareto_results_text": str(pareto_text_a),
        },
        expt_b.name: {
            "intersection_frontier_record": {
                "epoch": int(frontier_record_b["epoch"]),
                "step": int(frontier_record_b["step"]),
                "iteration": int(frontier_record_b["iteration"]),
                "recon_loss": float(frontier_record_b["recon_loss"]),
                "kl_loss": float(frontier_record_b["kl_loss"]),
            },
            "checkpoint": intersection_result_b["starting_checkpoint"],
            "distance_to_intersection_target": target["distance_b"],
            "rederived_state": intersection_rederived_state(
                intersection_result_b
            ),
            "rederived_weights": str(rederived_weights_b),
            "metrics": intersection_result_b["metrics"],
            "all_pareto_results_json": str(pareto_json_b),
            "all_pareto_results_text": str(pareto_text_b),
        },
    }

    if target["kind"] == "closest_approach":
        output["intersection"]["normalized_curve_distance"] = target["curve_distance"]

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    write_human_readable_report(
        report_path,
        args=args,
        output=output,
        expt_a_name=expt_a.name,
        expt_b_name=expt_b.name,
    )

    print()
    print("Intersection results")
    print("====================")
    for expt_name in (expt_a.name, expt_b.name):
        print(expt_name)
        for comparison, values in output[expt_name]["metrics"].items():
            print(
                f"  {comparison}: "
                f"FID={values['vae_feature_fid']:.8g}, "
                f"KID={values['vae_feature_kid']['mean']:.8g} "
                f"+/- {values['vae_feature_kid']['std']:.3g}"
            )

    print()
    print(f"Saved intersection JSON to {output_path}")
    print(f"Saved intersection text to {report_path}")
    print(f"Saved {expt_a.name} Pareto JSON to {pareto_json_a}")
    print(f"Saved {expt_a.name} Pareto text to {pareto_text_a}")
    print(f"Saved {expt_b.name} Pareto JSON to {pareto_json_b}")
    print(f"Saved {expt_b.name} Pareto text to {pareto_text_b}")

    total_elapsed = timedelta(seconds=perf_counter() - run_start)
    print(f"Total elapsed time: {total_elapsed}")


if __name__ == "__main__":
    main()
