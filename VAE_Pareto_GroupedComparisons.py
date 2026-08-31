#!/usr/bin/env python3
"""
Plot Pareto fronts for an inclusive range of VAE experiments, grouping curves
by fixed beta value or by greedy/adaptive beta.

Example
-------
python VAE_Pareto_GroupedComparisons.py \
    --first_expt 483 \
    --last_expt 532

All experiment directories from expt_483 through expt_532, inclusive, are
loaded. Each experiment contributes one Pareto-front curve.

Experiments with the same fixed beta value share one color and one legend
entry. All adaptive/greedy-beta experiments share one color and the legend
entry "Greedy beta".

Learning rate does not affect grouping or the legend.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from VAE_Anime_Config import TrainerConfig
from utils import get_experiment_dir


def resolve_experiment_dir(parent_dir, expt_num):
    expt_dir = get_experiment_dir(parent_dir, expt_num)

    if not expt_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment directory does not exist: {expt_dir}"
        )

    return expt_dir


def resolve_losses_file(expt_dir):
    losses_file = Path(expt_dir) / "stats" / "losses_file.txt"

    if not losses_file.is_file():
        raise FileNotFoundError(
            f"Loss file does not exist: {losses_file}"
        )

    return losses_file


def read_loss_records(expt_dir):
    """
    Read chronological loss records from stats/losses_file.txt.

    Iteration is the zero-based index of each valid recorded loss row.
    """
    losses_file = resolve_losses_file(expt_dir)
    records = []

    with losses_file.open("r", encoding="utf-8") as fh:
        next(fh, None)  # header

        for line_number, line in enumerate(fh, start=2):
            lower_line = line.lower()
            if "nan" in lower_line or "inf" in lower_line:
                break

            fields = [field.strip() for field in line.split("--")]

            if len(fields) < 4:
                print(
                    f"Warning: ignoring incomplete loss record at "
                    f"{losses_file}:{line_number}"
                )
                break

            try:
                epoch = int(fields[0])
                step = int(fields[1])
                recon_loss = float(fields[2])
                kl_loss = float(fields[3])
                kl_weight = float(fields[4].split(' ')[0])
            except (ValueError, IndexError):
                print(
                    f"Warning: ignoring malformed loss record at "
                    f"{losses_file}:{line_number}"
                )
                break

            if not (
                math.isfinite(recon_loss)
                and math.isfinite(kl_loss)
                and math.isfinite(kl_weight)
                and kl_loss > 0.0
            ):
                break

            records.append({
                "iteration": len(records),
                "epoch": epoch,
                "step": step,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "kl_weight": kl_weight,
            })

    if not records:
        raise ValueError(
            f"No valid loss records found in {losses_file}"
        )

    return records


def pareto_records(records):
    """
    Return records on the exact two-objective minimization frontier.

    This is the same frontier construction used by VAE_Pareto_Comparisons.py:
    sort by reconstruction loss, keep the lowest KL loss for duplicate
    reconstruction values, then retain only points that improve the best KL
    loss seen so far.
    """
    ordered = sorted(
        records,
        key=lambda record: (
            record["recon_loss"],
            record["kl_loss"],
        ),
    )

    collapsed = []

    for record in ordered:
        if (
            collapsed
            and record["recon_loss"]
            == collapsed[-1]["recon_loss"]
        ):
            if record["kl_loss"] < collapsed[-1]["kl_loss"]:
                collapsed[-1] = record
        else:
            collapsed.append(record)

    frontier = []
    best_kl = np.inf

    for record in collapsed:
        if record["kl_loss"] < best_kl:
            frontier.append(record)
            best_kl = record["kl_loss"]

    return frontier


def normalize_policy_name(loss_policy):
    if loss_policy is None:
        return ""

    return str(loss_policy).strip().lower().replace("-", "_")


def format_beta(beta):
    """
    Produce compact, stable beta labels.

    Examples:
        1.0    -> "1"
        10.0   -> "10"
        0.001  -> "0.001"
    """
    beta = float(beta)

    if beta == 0.0:
        return "0"

    if beta.is_integer():
        return str(int(beta))

    return f"{beta:g}"


def get_beta_group(expt_dir, records):
    """
    Determine an experiment's plotting group.

    config.ini is used only to identify fixed-beta versus adaptive/greedy-beta.
    For fixed-beta experiments, beta is read from the recorded kl_weight values
    in losses_file.txt, so grouping reflects what was actually run.
    """
    cfg_path = Path(expt_dir) / "config.ini"

    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Experiment config does not exist: {cfg_path}"
        )

    cfg = TrainerConfig.from_file(cfg_path)
    policy = normalize_policy_name(
        getattr(cfg, "loss_policy", None)
    )

    greedy_policy_names = {
        "adaptive_kl",
        "adaptive_beta",
        "greedy_beta",
        "greedy_kl",
    }

    fixed_policy_names = {
        "fixed_beta",
        "fixed_kl",
    }

    if policy in greedy_policy_names:
        return {
            "group_key": ("greedy", None),
            "group_label": "Greedy β",
            "loss_policy": policy,
            "beta": None,
        }

    if policy in fixed_policy_names:
        beta_values = np.asarray(
            [record["kl_weight"] for record in records],
            dtype=float,
        )

        beta = float(beta_values[0])

        if not np.allclose(
            beta_values,
            beta,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                f"{expt_dir.name}: fixed-beta run has non-constant "
                f"recorded kl_weight values"
            )

        return {
            "group_key": ("fixed", beta),
            "group_label": f"β = {format_beta(beta)}",
            "loss_policy": policy,
            "beta": beta,
        }

    raise ValueError(
        f"{expt_dir.name}: unrecognized loss_policy={policy!r}. "
        f"Expected fixed_beta or adaptive/greedy beta."
    )


def load_experiment(expt_num, parent_dir, skip_fraction):
    expt_dir = resolve_experiment_dir(parent_dir, expt_num)
    records = read_loss_records(expt_dir)

    skip_points = int(skip_fraction * len(records))
    retained_records = records[skip_points:]

    if not retained_records:
        raise ValueError(
            f"{expt_dir.name}: no records remain after "
            f"skip_fraction={skip_fraction}"
        )

    frontier_records = pareto_records(retained_records)

    if not frontier_records:
        raise ValueError(
            f"{expt_dir.name}: no Pareto points remain after "
            f"skip_fraction={skip_fraction}"
        )

    group = get_beta_group(expt_dir, records)

    return {
        "expt_num": expt_num,
        "dir": expt_dir,
        "records": records,
        "frontier_records": frontier_records,
        **group,
    }


def group_sort_key(group_key):
    kind, beta = group_key

    if kind == "fixed":
        return (0, float(beta))

    return (1, float("inf"))


def build_groups(experiments):
    groups = {}

    for expt in experiments:
        key = expt["group_key"]

        if key not in groups:
            groups[key] = {
                "label": expt["group_label"],
                "experiments": [],
            }

        groups[key]["experiments"].append(expt)

    return dict(
        sorted(
            groups.items(),
            key=lambda item: group_sort_key(item[0]),
        )
    )


def plot_grouped_pareto_curves(
    experiments,
    output_dir,
    first_expt,
    last_expt,
    skip_fraction,
    linewidth=1.15,
    alpha=0.60,
):
    """
    Plot every experiment's Pareto front, with one color/legend entry per beta
    group rather than one legend entry per experiment.
    """
    groups = build_groups(experiments)

    fig, ax = plt.subplots(
        figsize=(11, 8),
        constrained_layout=True,
    )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    legend_handles = []

    for group_index, (group_key, group) in enumerate(groups.items()):
        color = color_cycle[group_index % len(color_cycle)]

        for expt in group["experiments"]:
            frontier = expt["frontier_records"]

            curve = np.asarray([
                [
                    record["recon_loss"],
                    record["kl_loss"],
                ]
                for record in frontier
            ], dtype=float)

            if len(curve) <=2:
                print(f"Skipping {group_index}")
                continue

            ax.plot(
                curve[:, 0],
                curve[:, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.5,
                label=group["label"],
            )
        )

    ax.set_xlabel("Recon Loss")
    ax.set_ylabel("KL Loss")
    ax.set_yscale("log")
    ax.set_title(
        f"Sample Pareto Curves"
    )

    ax.legend(
        handles=legend_handles,
        title="Loss policy",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    outfile = output_dir / (
        f"ParetoCurves_Grouped_expt_{first_expt}_to_{last_expt}.png"
    )

    fig.savefig(outfile, dpi=200)
    plt.close(fig)

    print(f"Saved {outfile}")

    return outfile, groups


def save_group_report(
    experiments,
    groups,
    output_dir,
    first_expt,
    last_expt,
    skip_fraction,
):
    """
    Save a compact record of which experiment was assigned to which group.
    This is useful for catching configuration mistakes before interpreting the
    graph.
    """
    report = {
        "first_expt": first_expt,
        "last_expt": last_expt,
        "num_experiments": len(experiments),
        "skip_fraction": skip_fraction,
        "groups": [],
    }

    for group_key, group in groups.items():
        kind, beta = group_key

        report["groups"].append({
            "label": group["label"],
            "kind": kind,
            "beta": beta,
            "num_experiments": len(group["experiments"]),
            "experiments": [
                expt["expt_num"]
                for expt in group["experiments"]
            ],
        })

    outfile = output_dir / (
        f"ParetoCurves_Grouped_expt_{first_expt}_to_{last_expt}.json"
    )

    with outfile.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"Saved {outfile}")

    return outfile


def print_group_summary(groups):
    print()
    print("Experiment groups")
    print("=================")

    for _, group in groups.items():
        expt_nums = [
            expt["expt_num"]
            for expt in group["experiments"]
        ]

        print(
            f"{group['label']}: "
            f"{len(expt_nums)} experiments "
            f"(expt_{min(expt_nums)} to expt_{max(expt_nums)})"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Pareto fronts for every experiment in an inclusive "
            "experiment-number range, grouped by fixed beta or Greedy beta."
        )
    )

    parser.add_argument(
        "--first_expt",
        type=int,
        required=True,
        help="First experiment number in the inclusive range",
    )

    parser.add_argument(
        "--last_expt",
        type=int,
        required=True,
        help="Last experiment number in the inclusive range",
    )

    parser.add_argument(
        "--parent_dir",
        type=Path,
        default=Path("expts"),
        help=(
            "Parent directory containing expt_N directories "
            "(default: expts)"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory for outputs "
            "(default: <parent_dir>/pareto_comps)"
        ),
    )

    parser.add_argument(
        "--skip_fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of earliest chronological loss records to exclude "
            "before calculating each Pareto frontier "
            "(default: 0.10, matching VAE_Pareto_Comparisons.py)"
        ),
    )

    parser.add_argument(
        "--linewidth",
        type=float,
        default=1.15,
        help="Width of each individual Pareto curve (default: 1.15)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.60,
        help="Opacity of individual Pareto curves (default: 0.60)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.first_expt > args.last_expt:
        raise ValueError(
            "--first_expt must be less than or equal to --last_expt"
        )

    if not 0.0 <= args.skip_fraction < 1.0:
        raise ValueError(
            "--skip_fraction must satisfy 0 <= value < 1"
        )

    if args.linewidth <= 0.0:
        raise ValueError("--linewidth must be > 0")

    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("--alpha must satisfy 0 < value <= 1")

    expt_nums = range(
        args.first_expt,
        args.last_expt + 1,
    )

    experiments = []

    for expt_num in expt_nums:
        print(f"Loading expt_{expt_num}...")

        try:
            experiment = load_experiment(
                expt_num=expt_num,
                parent_dir=args.parent_dir,
                skip_fraction=args.skip_fraction,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(
                f"Warning: skipping expt_{expt_num}: {exc}"
            )
            continue

        experiments.append(experiment)

    if not experiments:
        raise RuntimeError(
            "No usable experiments were found in the requested range."
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.parent_dir / "pareto_comps"
    )

    _, groups = plot_grouped_pareto_curves(
        experiments=experiments,
        output_dir=output_dir,
        first_expt=args.first_expt,
        last_expt=args.last_expt,
        skip_fraction=args.skip_fraction,
        linewidth=args.linewidth,
        alpha=args.alpha,
    )

    save_group_report(
        experiments=experiments,
        groups=groups,
        output_dir=output_dir,
        first_expt=args.first_expt,
        last_expt=args.last_expt,
        skip_fraction=args.skip_fraction,
    )

    print_group_summary(groups)


if __name__ == "__main__":
    main()
