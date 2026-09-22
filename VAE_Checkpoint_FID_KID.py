"""Evaluate VAE-feature FID/KID at every periodic training checkpoint.

Purpose
-------
``VAE_Pareto_Intersection_FID_KID.py`` answers a point-selection question:
what did a VAE look like at an exact Pareto point or at the point on either
side of a Pareto-front intersection?  Those targets usually fall between
periodic checkpoints, so that module must restore a complete training state
and replay training to the exact recorded epoch and step.

This module answers a different, simpler question:

    How did generated-image quality change over training at the regularly
    saved model checkpoints?

Each target here *is* a saved checkpoint.  The corresponding ``.weights.h5``
file therefore contains the exact model parameters to evaluate; optimizer,
controller, and random-stream state are irrelevant because no training is
performed.  Avoiding replay makes a checkpoint-timeline scan both cheaper and
less fragile than exact reconstruction of arbitrary training steps.

Metrics
-------
The actual metric implementation is shared with
``VAE_Pareto_Intersection_FID_KID.py``.  It compares three distributions:

``generated_vs_raw``
    Images decoded from fixed N(0, I) samples versus real validation images.
    This is the primary generated-image-quality result.

``reconstructed_vs_raw``
    Deterministic decoder(mu(x)) reconstructions versus the real images.

``generated_vs_reconstructed``
    Prior-generated images versus deterministic reconstructions.  This is a
    useful diagnostic of the gap between the prior and aggregate posterior.

All images are embedded by one frozen reference-VAE encoder.  Consequently,
the reported values are deliberately called ``vae_feature_fid`` and
``vae_feature_kid``; they are not standard Inception FID/KID values.

Comparability and resumption
----------------------------
One invocation constructs the real-image set, reference features, FID raw
statistics, prior latent samples, and KID subset plan once.  Every checkpoint
and every requested experiment therefore receives the same evaluation inputs.

Results are written after every completed checkpoint so an interrupted scan
can resume.  Reuse is allowed only when an evaluation signature matches.  The
signature hashes the reference weights, reference config, and candidate
config, and also records every command-line option that changes the metric.
This prevents a partially completed JSON file from silently mixing results
from different feature extractors, sample populations, or KID settings.

Output
------
Each experiment writes:

    expts/expt_N/FID_KID/Checkpoint_FID_KID.json

The JSON is ordered chronologically and includes the checkpoint's original
training recon/KL values, KL weight, effective KL dimensions, and all three
FID/KID comparisons.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from VAE_Anime_ArtifactReader import ArtifactReader
from VAE_Pareto_Intersection_FID_KID import (
    build_vae_from_config,
    collect_images,
    evaluate_candidate,
    extract_features,
    feature_distribution_stats,
    find_checkpoint_records,
    load_reference_vae,
    make_reference_feature_model,
    make_reference_validation_dataset,
    resolve_experiment_dir,
)


# Increment this if the persisted JSON structure or resume semantics change in
# a way that makes old files unsafe to reuse.  This is intentionally separate
# from the evaluation signature, which detects changed model/metric inputs.
SCHEMA_VERSION = 1

# Keep checkpoint-timeline output distinct from Pareto_FID_KID.json.  The two
# files describe different sampling populations: regularly spaced saved states
# here, and reconstruction/KL Pareto states in the existing evaluator.
RESULT_FILE = "Checkpoint_FID_KID.json"


def sha256_file(path: Path) -> str:
    """Return a stable content fingerprint without loading a large file at once.

    Reference weights can be hundreds of megabytes, so the file is hashed in
    1 MiB chunks.  Content hashing is used instead of timestamps because copied
    files can retain or acquire misleading modification times.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_kl_dimensions(values) -> float:
    """Return the KL participation ratio for one training iteration.

    ``(sum KL_d)^2 / sum(KL_d^2)`` is near 1 when almost all KL is concentrated
    in one latent dimension and reaches ``latent_dim`` when KL is distributed
    equally across all dimensions.  Tiny negative values can arise from
    floating-point error even though analytical per-dimension KL is
    non-negative, so those values are clipped to zero before the calculation.

    A zero-KL vector has no participating dimensions and is defined as 0.0.
    """

    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    denominator = float(np.dot(values, values))
    if denominator == 0.0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def checkpoint_key(record: dict) -> str:
    """Build the canonical epoch/step identifier used by checkpoint files."""

    return (
        f"epoch_{int(record['epoch']):04d}_"
        f"step_{int(record['step']):04d}"
    )


def weight_checkpoint_path(expt_dir: Path, record: dict) -> Path:
    """Map a matched full-state checkpoint record to its weight-only file.

    Training writes two artifacts at the same epoch boundary:

    * ``checkpoints/epoch_XXXX_step_XXXX.weights.h5`` for direct inference;
    * ``checkpoints/training_state/epoch_XXXX_step_XXXX/`` for exact resume.

    ``find_checkpoint_records`` discovers and validates the latter because it
    also matches epoch/step to the loss history.  Evaluation loads the former
    because inference needs only model weights and is faster to initialize.
    """

    return (
        expt_dir
        / "checkpoints"
        / f"{checkpoint_key(record)}.weights.h5"
    )


def make_signature(*, args, ref_dir: Path, candidate_config: Path) -> dict:
    """Describe every input whose change makes cached metrics incompatible."""

    # Include the candidate config even though many settings do not directly
    # affect inference.  It is cheap to hash and ensures cached values cannot
    # be attached to a silently changed model architecture or experiment.
    return {
        "reference_experiment": int(args.ref_expt),
        "reference_config_sha256": sha256_file(ref_dir / "config.ini"),
        "reference_weights_sha256": sha256_file(
            ref_dir / "reference_vae.weights.h5"
        ),
        "candidate_config_sha256": sha256_file(candidate_config),
        "feature_layer": args.feature_layer,
        "n_images": int(args.n_images),
        "batch_size": int(args.batch_size),
        "kid_subset_size": int(args.kid_subset_size),
        "kid_subsets": int(args.kid_subsets),
        "eval_seed": int(args.eval_seed),
    }


def load_progress(path: Path, signature: dict, *, overwrite: bool) -> dict:
    """Load completed checkpoint rows after proving they are compatible.

    Returning a dictionary keyed by checkpoint name makes the hot-path resume
    check O(1).  A malformed JSON file is deliberately an error rather than an
    empty cache: silently restarting could overwrite evidence of disk damage or
    an interrupted non-atomic writer from an older implementation.

    ``--overwrite`` bypasses reading entirely.  This is the explicit escape
    hatch for intentionally changing the evaluation definition.
    """

    if overwrite or not path.is_file():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible checkpoint FID/KID schema in {path}; "
            "use --overwrite to replace it"
        )
    if payload.get("evaluation_signature") != signature:
        raise ValueError(
            f"Cached checkpoint metrics in {path} use different evaluation "
            "settings or model files; use --overwrite to replace them"
        )

    return {
        item["checkpoint_key"]: item
        for item in payload.get("checkpoints", [])
        if "checkpoint_key" in item and "metrics" in item
    }


def save_progress(
    path: Path,
    *,
    experiment: str,
    signature: dict,
    results_by_key: dict,
    status: str,
):
    """Atomically persist all completed checkpoint evaluations.

    The output is rewritten after each checkpoint because one metric
    calculation is expensive relative to this small JSON write.  Writing a
    complete temporary file and then replacing the destination ensures an
    interruption leaves either the previous valid version or the new valid
    version, not a half-written JSON document.

    ``status`` remains ``in_progress`` until every discovered checkpoint has
    been handled.  Consumers can therefore distinguish a resumable partial
    scan from a complete timeline.
    """

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "status": status,
        "evaluation_signature": signature,
        "metric_definition": {
            "feature_space": "fixed reference-VAE encoder",
            "fid_name": "vae_feature_fid",
            "kid_name": "vae_feature_kid",
            "note": "These are not standard Inception FID/KID values.",
            "generated_latents": (
                "the same fixed N(0,I) samples are used at every checkpoint"
            ),
        },
        "checkpoints": sorted(
            results_by_key.values(),
            key=lambda item: int(item["iteration"]),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def evaluate_experiment(
    *,
    expt_dir: Path,
    args,
    ref_dir: Path,
    feature_model,
    raw_images: np.ndarray,
    raw_features: np.ndarray,
    raw_stats,
):
    """Evaluate all periodic checkpoints belonging to one experiment.

    Shared reference-side inputs are supplied by ``main`` so they are computed
    once for the entire multi-experiment invocation.  Candidate weights are
    loaded successively into one reconstructed architecture; rebuilding a new
    Keras model for every checkpoint would add overhead without changing the
    result.
    """

    config_path = expt_dir / "config.ini"
    vae, cfg = build_vae_from_config(config_path, expt_dir / "model_info")
    if cfg.image_size != raw_images.shape[1]:
        raise ValueError(
            f"{expt_dir.name}: candidate and reference image sizes differ"
        )

    signature = make_signature(
        args=args,
        ref_dir=ref_dir,
        candidate_config=config_path,
    )
    output_path = expt_dir / "FID_KID" / RESULT_FILE
    results_by_key = load_progress(
        output_path,
        signature,
        overwrite=args.overwrite,
    )

    # This returns only complete full-state checkpoints that match an exact
    # epoch/step entry in losses_file.txt.  Although we load the companion
    # weight-only file below, using this validated record set ensures each row
    # has trustworthy chronology and training-loss metadata.
    checkpoints = find_checkpoint_records(expt_dir)

    # latent_kl_stats.pkl is recorded once per accepted training iteration in
    # the same order used by read_loss_records.  The checkpoint iteration is
    # therefore the direct index for its per-dimension KL vector.
    latent_kl = ArtifactReader(expt_dir / "stats").read_latent_kl_series().kl_per_dim
    max_iteration = max(int(record["iteration"]) for record in checkpoints)
    if max_iteration >= len(latent_kl):
        raise ValueError(
            f"{expt_dir.name}: checkpoint iteration {max_iteration} has no "
            "matching latent KL record"
        )

    # Recreate the same latent sample array independently for each experiment.
    # Resetting the generator, rather than allowing it to advance between
    # experiments, is what guarantees point-for-point comparability.
    rng = np.random.default_rng(args.eval_seed)
    latent_samples = rng.standard_normal(
        (args.n_images, cfg.latent_dim)
    ).astype(np.float32)

    total = len(checkpoints)
    for index, record in enumerate(checkpoints, start=1):
        key = checkpoint_key(record)
        if key in results_by_key:
            print(f"{expt_dir.name}: {key} already evaluated; skipping")
            continue

        weights_path = weight_checkpoint_path(expt_dir, record)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Weight checkpoint does not exist: {weights_path}"
            )

        started = datetime.now()
        print(
            f"{expt_dir.name}: checkpoint {index}/{total} {key} "
            f"started {started:%Y-%m-%d %H:%M:%S}"
        )
        # The checkpoint is itself the target.  Loading weights is sufficient:
        # no optimizer/controller restoration or forward training replay is
        # needed for deterministic reconstruction and prior generation.
        vae.vae_net.load_weights(weights_path)
        metrics = evaluate_candidate(
            vae,
            raw_images,
            latent_samples,
            feature_model,
            batch_size=args.batch_size,
            raw_features=raw_features,
            raw_stats=raw_stats,
            kid_subset_size=args.kid_subset_size,
            kid_subsets=args.kid_subsets,
            kid_seed=args.eval_seed + 100,
        )

        iteration = int(record["iteration"])
        results_by_key[key] = {
            "checkpoint_key": key,
            "weights": str(weights_path),
            "epoch": int(record["epoch"]),
            "step": int(record["step"]),
            "iteration": iteration,
            "recon_loss": float(record["recon_loss"]),
            "kl_loss": float(record["kl_loss"]),
            "kl_weight": (
                None
                if record.get("kl_weight") is None
                else float(record["kl_weight"])
            ),
            "effective_kl_dimensions": effective_kl_dimensions(
                latent_kl[iteration]
            ),
            "metrics": metrics,
        }
        # Commit immediately after the expensive metric calculation.  A crash
        # during the following checkpoint loses at most that unfinished point.
        save_progress(
            output_path,
            experiment=expt_dir.name,
            signature=signature,
            results_by_key=results_by_key,
            status="in_progress",
        )

        gen_raw = metrics["generated_vs_raw"]
        print(
            f"  FID={gen_raw['vae_feature_fid']:.6g}, "
            f"KID={gen_raw['vae_feature_kid']['mean']:.6g}"
        )

    save_progress(
        output_path,
        experiment=expt_dir.name,
        signature=signature,
        results_by_key=results_by_key,
        status="complete",
    )
    print(f"Saved {output_path}")


def parse_args(argv=None):
    """Parse CLI arguments; accepting argv also makes parser tests easy."""

    parser = argparse.ArgumentParser(
        description="Evaluate VAE-feature FID/KID at every saved checkpoint"
    )
    parser.add_argument("--expts", type=int, nargs="+", required=True)
    parser.add_argument("--ref_expt", type=int, required=True)
    parser.add_argument("--parent_dir", type=Path, default=Path("expts"))
    parser.add_argument("--ref_root", type=Path, default=Path("ref_vae"))
    parser.add_argument("--feature_layer", default="lrelu_4")
    parser.add_argument("--n_images", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--kid_subsets", type=int, default=20)
    parser.add_argument("--eval_seed", type=int, default=12345)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard compatible or incompatible saved checkpoint metrics",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Build common evaluation inputs once, then scan requested experiments."""

    run_start = perf_counter()
    args = parse_args(argv)
    if args.n_images < 2:
        raise ValueError("--n_images must be at least 2")
    if not 2 <= args.kid_subset_size <= args.n_images:
        raise ValueError(
            "--kid_subset_size must be between 2 and --n_images"
        )
    if args.kid_subsets <= 0:
        raise ValueError("--kid_subsets must be > 0")

    # The reference encoder, raw images, and raw-image features are invariant
    # across all candidate checkpoints.  Constructing them outside the
    # experiment loop both saves work and enforces a common comparison space.
    ref_vae, ref_cfg, ref_dir = load_reference_vae(
        args.ref_root,
        args.ref_expt,
    )
    feature_model = make_reference_feature_model(ref_vae, args.feature_layer)
    validation_dataset = make_reference_validation_dataset(
        ref_cfg,
        scratch_dir=ref_dir / "_eval_dataset",
    )
    raw_images = collect_images(validation_dataset, args.n_images)
    raw_features = extract_features(feature_model, raw_images, args.batch_size)
    raw_stats = feature_distribution_stats(raw_features)

    for expt_num in args.expts:
        evaluate_experiment(
            expt_dir=resolve_experiment_dir(args.parent_dir, expt_num),
            args=args,
            ref_dir=ref_dir,
            feature_model=feature_model,
            raw_images=raw_images,
            raw_features=raw_features,
            raw_stats=raw_stats,
        )

    print(f"Total elapsed time: {perf_counter() - run_start:.1f} seconds")


if __name__ == "__main__":
    main()
