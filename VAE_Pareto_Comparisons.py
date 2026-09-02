import argparse
import json
import numpy as np
import re
import matplotlib.pyplot as plt

from pathlib import Path

from VAE_Anime_Config import TrainerConfig
from VAE_ParetoFront import ParetoFront
from VAE_Loss_Records import (
    read_loss_records as shared_read_loss_records,
    pareto_records,
)

from utils import get_experiment_dir


def sanitize_filename_part(text):
    text = str(text)
    text = re.sub(r'[<>:"/\\|?*]+', '_', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:180]


def get_experiment_label(expt_dir, fallback_label=None):
    expt_dir = Path(expt_dir)
    cfg_path = expt_dir / "config.ini"

    if cfg_path.is_file():
        cfg = TrainerConfig.from_file(cfg_path)
        if cfg.expt_name:
            return f"{expt_dir.name}: {cfg.expt_name}"

    if fallback_label is not None:
        return f"{expt_dir.name}: {fallback_label}"

    return expt_dir.name


def read_loss_records(expt_dir):
    """Read chronological loss records for an experiment directory."""
    return shared_read_loss_records(resolve_losses_file(expt_dir))


def resolve_experiment_dir(parent_dir, expt_num=None, expt_dir=None):
    if expt_dir is not None:
        resolved = Path(expt_dir)
    elif expt_num is not None:
        resolved = get_experiment_dir(parent_dir, expt_num)
    else:
        raise ValueError("Must provide either experiment number or experiment directory")

    if not resolved.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {resolved}")
    return resolved


def resolve_losses_file(expt_dir):
    losses_file = Path(expt_dir) / "stats" / "losses_file.txt"
    if not losses_file.is_file():
        raise FileNotFoundError(f"Loss file does not exist: {losses_file}")
    return losses_file


def build_experiment_specs(parent_dir, expt_nums=None, expt_dirs=None):
    """
    Build a list of experiment specs:
        [{"dir": Path(...), "fallback_label": ...}, ...]
    """
    expt_nums = expt_nums or []
    expt_dirs = expt_dirs or []

    if not expt_nums and not expt_dirs:
        raise ValueError("Provide at least one experiment via --expts / --expt_dirs "
                         "or via legacy --expt1/--expt2 options.")

    specs = []

    for expt_num in expt_nums:
        expt_dir = resolve_experiment_dir(parent_dir, expt_num=expt_num)
        specs.append({
            "dir": expt_dir,
            "fallback_label": str(expt_num),
        })

    for expt_dir_str in expt_dirs:
        expt_dir = resolve_experiment_dir(parent_dir, expt_dir=expt_dir_str)
        specs.append({
            "dir": expt_dir,
            "fallback_label": None,
        })

    return specs


def load_experiments(specs):
    experiments = []

    for spec in specs:
        expt_dir = spec["dir"]
        fallback_label = spec["fallback_label"]

        records = read_loss_records(expt_dir)
        label = get_experiment_label(expt_dir, fallback_label=fallback_label)

        experiments.append({
            "dir": expt_dir,
            "label": label,
            "records": records,
            "recon_pts": [record["recon_loss"] for record in records],
            "kl_pts": [record["kl_loss"] for record in records],
        })

    return experiments


def make_output_stem(experiments, prefix, max_labels=6):
    labels = [sanitize_filename_part(expt["dir"].name) for expt in experiments[:max_labels]]
    if len(experiments) > max_labels:
        labels.append(f"plus_{len(experiments) - max_labels}_more")
    return f"{prefix} {'__'.join(labels)}"


def compare_graphs(experiments, output_dir, skip_fraction=0.10):
    if len(experiments) < 1:
        raise ValueError("Need at least one experiment to plot")

    fig, ax = plt.subplots(2, figsize=(10, 10), constrained_layout=True)

    for idx, expt in enumerate(experiments):
        recon_pts = expt["recon_pts"]
        kl_pts = expt["kl_pts"]
        label = expt["label"]

        skip_pts = int(skip_fraction * len(recon_pts))
        x = recon_pts[skip_pts:]
        y = kl_pts[skip_pts:]

        ax[0].scatter(x, y, s=1, label=label)

    ax[0].set_title("Pareto-ish Graph")
    ax[0].set_xlabel("Recon Loss")
    ax[0].set_ylabel("KL Loss")
    ax[0].set_yscale("log")
    ax[0].legend(markerscale=4)

    # Reverse plot order on second axes to reduce overplotting bias
    for expt in reversed(experiments):
        recon_pts = expt["recon_pts"]
        kl_pts = expt["kl_pts"]
        label = expt["label"]

        skip_pts = int(skip_fraction * len(recon_pts))
        x = recon_pts[skip_pts:]
        y = kl_pts[skip_pts:]

        ax[1].scatter(x, y, s=1, label=label)

    ax[1].set_xlabel("Recon Loss")
    ax[1].set_ylabel("KL Loss")
    ax[1].set_yscale("log")
    ax[1].legend(markerscale=4)

    outfile = output_dir / f"{make_output_stem(experiments, 'Pareto_Comparisons')}.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


def compare_pareto_curves(experiments, output_dir, skip_fraction=0.10):
    if len(experiments) < 1:
        raise ValueError("Need at least one experiment to plot")

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    reports = []

    for expt in experiments:
        records = expt["records"]
        label = expt["label"]
        skip_pts = int(skip_fraction * len(records))
        retained_records = records[skip_pts:]
        frontier_records = pareto_records(retained_records)

        if not frontier_records:
            print(f"Warning: no Pareto points retained for {label}")
            continue

        pareto_curve = np.asarray([
            [record["recon_loss"], record["kl_loss"]]
            for record in frontier_records
        ], dtype=float)

        ax.plot(
            pareto_curve[:, 0],
            pareto_curve[:, 1],
            linewidth=2,
            label=label,
        )
        ax.scatter(pareto_curve[:, 0], pareto_curve[:, 1], s=10)

        chronological_points = sorted(
            frontier_records,
            key=lambda record: record["iteration"],
        )
        last_pareto_point = chronological_points[-1]
        final_training_point = records[-1]

        reports.append({
            "experiment": expt["dir"].name,
            "label": label,
            "skip_fraction": skip_fraction,
            "num_training_points": len(records),
            "num_retained_training_points": len(retained_records),
            "num_pareto_points": len(chronological_points),
            "final_training_epoch": final_training_point["epoch"],
            "final_training_step": final_training_point["step"],
            "final_training_iteration": final_training_point["iteration"],
            "last_pareto_epoch": last_pareto_point["epoch"],
            "last_pareto_step": last_pareto_point["step"],
            "last_pareto_iteration": last_pareto_point["iteration"],
            "pareto_training_fraction": (
                (last_pareto_point["iteration"] + 1) /
                (final_training_point["iteration"] + 1)
            ),
            "last_pareto_point": last_pareto_point,
            "pareto_points": chronological_points,
        })

        print(
            f"{expt['dir'].name}: {len(chronological_points)} Pareto points; "
            f"last Pareto point at epoch {last_pareto_point['epoch']}, "
            f"step {last_pareto_point['step']}, "
            f"iteration {last_pareto_point['iteration']} "
            f"({reports[-1]['pareto_training_fraction']:.1%} of recorded training)"
        )

    ax.set_xlabel("Recon Loss")
    ax.set_ylabel("KL Loss")
    ax.set_yscale("log")
    ax.set_title("Pareto Curves")
    ax.legend()

    outfile = output_dir / f"{make_output_stem(experiments, 'ParetoCurves')}.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")

    report_file = output_dir / (
        f"{make_output_stem(experiments, 'ParetoIterations')}.json"
    )
    with report_file.open("w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    print(f"Saved {report_file}")


def load_curve(path):
    """
    Load a two-column text file with optional header:
        recon_loss kl_loss
        ...
    """
    return np.loadtxt(path, skiprows=1)


def closest_points_between_curves(path_a, path_b, use_log_kl=True):
    curve_a = load_curve(path_a)
    curve_b = load_curve(path_b)

    a_recon = curve_a[:, 0]
    a_kl = curve_a[:, 1]

    b_recon = curve_b[:, 0]
    b_kl = curve_b[:, 1]

    if use_log_kl:
        if np.any(a_kl <= 0) or np.any(b_kl <= 0):
            raise ValueError("KL losses must be positive when using log(KL).")

        a_kl_metric = np.log10(a_kl)
        b_kl_metric = np.log10(b_kl)
    else:
        a_kl_metric = a_kl
        b_kl_metric = b_kl

    # Combine both curves to define a shared scale.
    all_recon = np.concatenate([a_recon, b_recon])
    all_kl_metric = np.concatenate([a_kl_metric, b_kl_metric])

    recon_scale = all_recon.max() - all_recon.min()
    kl_scale = all_kl_metric.max() - all_kl_metric.min()

    if recon_scale == 0:
        recon_scale = 1.0
    if kl_scale == 0:
        kl_scale = 1.0

    a_scaled = np.column_stack([
        (a_recon - all_recon.min()) / recon_scale,
        (a_kl_metric - all_kl_metric.min()) / kl_scale,
    ])

    b_scaled = np.column_stack([
        (b_recon - all_recon.min()) / recon_scale,
        (b_kl_metric - all_kl_metric.min()) / kl_scale,
    ])

    # Pairwise squared distances: shape = (len(curve_a), len(curve_b))
    diff = a_scaled[:, None, :] - b_scaled[None, :, :]
    dist2 = np.sum(diff**2, axis=2)

    i, j = np.unravel_index(np.argmin(dist2), dist2.shape)

    return {
        f"index_{path_a.parent.parent.name}": i,
        f"index_{path_b.parent.parent.name}": j,
        f"point_{path_a.parent.parent.name}": curve_a[i],
        f"point_{path_b.parent.parent.name}": curve_b[j],
        "scaled_distance": np.sqrt(dist2[i, j]),
        "raw_recon_difference": abs(curve_a[i, 0] - curve_b[j, 0]),
        "raw_kl_difference": abs(curve_a[i, 1] - curve_b[j, 1]),
    }

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_closest_points_result(result, outfile):
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=to_jsonable)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Pareto-ish scatter plots and Pareto curves for any number of experiments."
    )

    # New multi-experiment interface
    parser.add_argument(
        "--expts",
        type=int,
        nargs="*",
        default=None,
        help="Experiment numbers, e.g. --expts 101 102 103",
    )
    parser.add_argument(
        "--expt_dirs",
        type=str,
        nargs="*",
        default=None,
        help="Experiment directories, e.g. --expt_dirs expts/expt_101 /tmp/expt_foo",
    )

    # Legacy two-experiment interface kept for compatibility
    parser.add_argument("--expt1", type=int, help="Number of first experiment")
    parser.add_argument("--expt2", type=int, help="Number of second experiment")
    parser.add_argument("--expt1_dir", type=str, default=None,
                        help="Path to first experiment directory")
    parser.add_argument("--expt2_dir", type=str, default=None,
                        help="Path to second experiment directory")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for comparison outputs")
    parser.add_argument("--skip_fraction", type=float, default=0.10,
                        help="Fraction of earliest points to skip")
    parser.add_argument(
        "--parent_dir",
        type=Path,
        default=Path("expts"),
        help="Parent directory containing experiment directories (default: expts)",
    )

    args = parser.parse_args()

    parent_dir = args.parent_dir

    expt_nums = list(args.expts) if args.expts else []
    expt_dirs = list(args.expt_dirs) if args.expt_dirs else []

    # Fold legacy args into the new lists
    if args.expt1 is not None:
        expt_nums.append(args.expt1)
    if args.expt2 is not None:
        expt_nums.append(args.expt2)
    if args.expt1_dir is not None:
        expt_dirs.append(args.expt1_dir)
    if args.expt2_dir is not None:
        expt_dirs.append(args.expt2_dir)

    specs = build_experiment_specs(parent_dir, expt_nums=expt_nums, expt_dirs=expt_dirs)
    experiments = load_experiments(specs)

    output_dir = Path(args.output_dir) if args.output_dir else parent_dir / "pareto_comps"
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_graphs(experiments, output_dir, skip_fraction=args.skip_fraction)
    compare_pareto_curves(experiments, output_dir, skip_fraction=args.skip_fraction)

    if len(expt_nums) == 2:        
        paretoPointFile = make_output_stem(experiments, 'ParetoPoints')
        outfile = output_dir / Path(paretoPointFile)
        outfile = outfile.with_suffix('.json')

        path0 = specs[0]['dir'] / Path("stats/ParetoPoints.txt")
        path1 = specs[1]['dir'] / Path("stats/ParetoPoints.txt")
        closestPointDict = closest_points_between_curves(path0, path1)
        save_closest_points_result(closestPointDict, outfile)
        print(f"Saved {outfile}")
