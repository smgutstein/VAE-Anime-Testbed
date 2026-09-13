"""Plot FID/KID relationships for already-evaluated recon/KL Pareto points."""

import argparse
import configparser
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from VAE_Anime_ArtifactReader import ArtifactReader


def effective_kl_dimensions(kl_per_dim):
    """KL participation ratio: (sum KL)^2 / sum(KL^2)."""
    values = np.asarray(kl_per_dim, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("kl_per_dim must be a non-empty finite one-dimensional array")
    # Per-dimension KL is non-negative analytically. Remove only numerical
    # undershoot rather than allowing it to inflate or cancel the total.
    values = np.maximum(values, 0.0)
    denominator = float(np.dot(values, values))
    if denominator == 0.0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def minimization_frontier(x_values, y_values):
    """Indices on the exact two-objective minimization frontier, sorted by x."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("Frontier coordinates must have matching shapes")

    valid = np.flatnonzero(np.isfinite(x_values) & np.isfinite(y_values))
    ordered = sorted(valid, key=lambda idx: (x_values[idx], y_values[idx], idx))
    frontier = []
    best_y = math.inf
    last_x = None

    for idx in ordered:
        x_value = x_values[idx]
        if last_x is not None and x_value == last_x:
            continue
        last_x = x_value
        if y_values[idx] < best_y:
            frontier.append(idx)
            best_y = y_values[idx]
    return frontier


def _experiment_label(expt_dir):
    config_path = expt_dir / "config.ini"
    parser = configparser.ConfigParser()
    if config_path.is_file():
        parser.read(config_path)
        for section in parser.sections():
            if parser.has_option(section, "expt_name"):
                name = parser.get(section, "expt_name").strip()
                if name:
                    return f"{expt_dir.name}: {name}"
    return expt_dir.name


def load_experiment(expt_dir):
    expt_dir = Path(expt_dir)
    results_path = expt_dir / "FID_KID" / "Pareto_FID_KID.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"FID/KID results do not exist: {results_path}")

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"FID/KID results are not complete: {results_path}")
    points = payload.get("pareto_points", [])
    if not points:
        raise ValueError(f"No Pareto points found in {results_path}")

    points = sorted(points, key=lambda item: int(item["iteration"]))
    latent_kl = ArtifactReader(expt_dir / "stats").read_latent_kl_series().kl_per_dim
    max_iteration = max(int(item["iteration"]) for item in points)
    if max_iteration >= len(latent_kl):
        raise ValueError(
            f"{expt_dir.name}: Pareto iteration {max_iteration} has no matching "
            f"entry in latent_kl_stats.pkl ({len(latent_kl)} entries)"
        )

    rows = []
    for item in points:
        metrics = item["metrics"]["generated_vs_raw"]
        iteration = int(item["iteration"])
        rows.append({
            "epoch": int(item["epoch"]),
            "step": int(item["step"]),
            "iteration": iteration,
            "recon": float(item["recon_loss"]),
            "kl": float(item["kl_loss"]),
            "fid": float(metrics["vae_feature_fid"]),
            "kid": float(metrics["vae_feature_kid"]["mean"]),
            "kid_std": float(metrics["vae_feature_kid"]["std"]),
            "effective_dims": effective_kl_dimensions(latent_kl[iteration]),
        })

    return {
        "dir": expt_dir,
        "label": _experiment_label(expt_dir),
        "rows": rows,
        "metric_definition": payload.get("metric_definition", {}),
    }


def _values(experiment, field):
    return np.asarray([row[field] for row in experiment["rows"]], dtype=float)


def chronological_colors(color, count, minimum_intensity=0.25):
    """Blend one experiment hue from pale (early) to full color (late)."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.empty((0, 3), dtype=float)
    base = np.asarray(to_rgb(color), dtype=float)
    intensity = np.linspace(minimum_intensity, 1.0, count)[:, None]
    return 1.0 - intensity * (1.0 - base)


def _plot_trace(ax, experiment, x_field, y_field, color, *, frontier=False):
    x_values = _values(experiment, x_field)
    y_values = _values(experiment, y_field)
    point_colors = chronological_colors(color, len(x_values))
    ax.scatter(
        x_values,
        y_values,
        s=14,
        c=point_colors,
        edgecolors="none",
    )
    if frontier:
        indices = minimization_frontier(x_values, y_values)
        ax.scatter(
            x_values[indices],
            y_values[indices],
            s=34,
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
        )


def _finish_axes(axes, experiments, colors):
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="none",
            label=experiment["label"],
        )
        for experiment, color in zip(experiments, colors)
    ]
    for ax in np.asarray(axes).flat:
        ax.grid(True, alpha=0.25)
        ax.legend(handles=handles)


def make_primary_figure(experiments, colors, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = (
        ("recon", "kl", "Original Recon/KL Pareto Points", "Reconstruction Loss", "KL Loss", False),
        ("recon", "fid", "Recon vs Generated/Raw FID", "Reconstruction Loss", "VAE-feature FID", True),
        ("recon", "kid", "Recon vs Generated/Raw KID", "Reconstruction Loss", "VAE-feature KID (mean)", True),
        ("fid", "kid", "Generated/Raw FID vs KID", "VAE-feature FID", "VAE-feature KID (mean)", False),
    )
    for ax, panel in zip(axes.flat, panels):
        x_field, y_field, title, xlabel, ylabel, frontier = panel
        for experiment, color in zip(experiments, colors):
            _plot_trace(ax, experiment, x_field, y_field, color, frontier=frontier)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axes[0, 0].set_yscale("log")
    _finish_axes(axes, experiments, colors)
    fig.suptitle("FID/KID at Original Recon/KL Pareto Points")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_kl_dimensions_figure(experiments, colors, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = (
        ("kl", "fid", "KL vs Generated/Raw FID", "KL Loss", "VAE-feature FID"),
        ("kl", "kid", "KL vs Generated/Raw KID", "KL Loss", "VAE-feature KID (mean)"),
        ("effective_dims", "fid", "Effective KL Dimensions vs FID", "Effective KL Dimensions", "VAE-feature FID"),
        ("effective_dims", "kid", "Effective KL Dimensions vs KID", "Effective KL Dimensions", "VAE-feature KID (mean)"),
    )
    for ax, panel in zip(axes.flat, panels):
        x_field, y_field, title, xlabel, ylabel = panel
        for experiment, color in zip(experiments, colors):
            _plot_trace(ax, experiment, x_field, y_field, color)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axes[0, 0].set_xscale("log")
    axes[0, 1].set_xscale("log")
    _finish_axes(axes, experiments, colors)
    fig.suptitle("Generation Quality vs KL and Effective KL Dimensions")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_iteration_figure(experiments, colors, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    panels = (
        ("iteration", "fid", "Generated/Raw FID vs Iteration", "VAE-feature FID"),
        ("iteration", "kid", "Generated/Raw KID vs Iteration", "VAE-feature KID (mean)"),
    )
    for ax, panel in zip(axes, panels):
        x_field, y_field, title, ylabel = panel
        for experiment, color in zip(experiments, colors):
            _plot_trace(ax, experiment, x_field, y_field, color)
        ax.set(title=title, xlabel="Iteration", ylabel=ylabel)
    _finish_axes(axes, experiments, colors)
    fig.suptitle("Generation Quality Across Original Pareto Iterations")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _output_stem(experiments):
    names = [re.sub(r"[^A-Za-z0-9_.-]+", "_", item["dir"].name) for item in experiments]
    return "__".join(names)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot saved VAE-feature FID/KID results at Pareto points"
    )
    parser.add_argument("--expts", type=int, nargs="+", required=True)
    parser.add_argument("--parent_dir", type=Path, default=Path("expts"))
    parser.add_argument(
        "--output_dir", type=Path, default=Path("pareto_comps") / "FID_KID"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = [
        load_experiment(args.parent_dir / f"expt_{number}")
        for number in args.expts
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [
        color_cycle[index % len(color_cycle)]
        for index in range(len(experiments))
    ]
    stem = _output_stem(experiments)
    outputs = (
        args.output_dir / f"FID_KID_Pareto_Overview__{stem}.png",
        args.output_dir / f"FID_KID_KL_EffectiveDims__{stem}.png",
        args.output_dir / f"FID_KID_Iterations__{stem}.png",
    )
    make_primary_figure(experiments, colors, outputs[0])
    make_kl_dimensions_figure(experiments, colors, outputs[1])
    make_iteration_figure(experiments, colors, outputs[2])
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
