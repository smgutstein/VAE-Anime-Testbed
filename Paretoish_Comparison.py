import argparse
import re
import matplotlib.pyplot as plt

from pathlib import Path

from VAE_Anime_ArtifactReader import ArtifactReader
from VAE_Anime_Config import TrainerConfig
from VAE_Anime_ResultsIO import read_loss_file_points_io
from VAE_ParetoFront import ParetoFront

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


def read_pareto_points(expt_dir):
    stats_dir = Path(expt_dir) / "stats"
    reader = ArtifactReader(stats_dir)

    try:
        series = reader.read_loss_series()
        return series.recon_loss, series.kl_loss
    except Exception:
        losses_file = resolve_losses_file(expt_dir)
        _, recon_pts, kl_pts, _ = read_loss_file_points_io(losses_file)
        return recon_pts, kl_pts


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

        recon_pts, kl_pts = read_pareto_points(expt_dir)
        label = get_experiment_label(expt_dir, fallback_label=fallback_label)

        experiments.append({
            "dir": expt_dir,
            "label": label,
            "recon_pts": recon_pts,
            "kl_pts": kl_pts,
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

    outfile = output_dir / f"{make_output_stem(experiments, 'Paretoish')}.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


def compare_pareto_curves(experiments, output_dir, skip_fraction=0.10):
    if len(experiments) < 1:
        raise ValueError("Need at least one experiment to plot")

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    for expt in experiments:
        recon_pts = expt["recon_pts"]
        kl_pts = expt["kl_pts"]
        label = expt["label"]

        skip_pts = int(skip_fraction * len(recon_pts))

        p = ParetoFront()
        p.add_points(recon_pts[skip_pts:], kl_pts[skip_pts:])

        # Use raw frontier, not smoothing, unless you have a good reason not to.
        pareto_curve = p.get_curve()

        if pareto_curve.size == 0:
            print(f"Warning: no Pareto points retained for {label}")
            continue

        ax.plot(pareto_curve[:, 0], pareto_curve[:, 1],
                linewidth=2, label=label)
        ax.scatter(pareto_curve[:, 0], pareto_curve[:, 1], s=10)

    ax.set_xlabel("Recon Loss")
    ax.set_ylabel("KL Loss")
    ax.set_yscale("log")
    ax.set_title("Pareto Curves")
    ax.legend()

    outfile = output_dir / f"{make_output_stem(experiments, 'ParetoCurves')}.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")


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
    parser.add_argument("-c", "--config_file", type=str, nargs="?",
                        default="config.ini", help="Config file")

    args = parser.parse_args()

    cfg = TrainerConfig.from_file(args.config_file)
    parent_dir = Path(cfg.parent_dir)

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