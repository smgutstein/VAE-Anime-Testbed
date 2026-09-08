#!/usr/bin/env python3
"""Render cumulative evolution of one or more empirical Pareto frontiers."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

from VAE_Loss_Records import pareto_records
from VAE_Pareto_GroupedComparisons_beta_lr import (
    build_groups,
    load_experiment,
)
from VAE_Pareto_Plotting import (
    configure_pareto_axes,
    plot_grouped_frontiers,
    resolve_axis_limits,
    validate_axis_limits,
)


def frame_iterations(experiments, frame_every_iterations):
    first_iteration = min(
        experiment["retained_records"][0]["iteration"]
        for experiment in experiments
    )
    final_iteration = max(
        experiment["retained_records"][-1]["iteration"]
        for experiment in experiments
    )
    frames = list(range(first_iteration, final_iteration + 1, frame_every_iterations))
    if frames[-1] != final_iteration:
        frames.append(final_iteration)
    return frames


def frontier_through_iteration(records, iteration):
    return pareto_records(
        [record for record in records if record["iteration"] <= iteration]
    )


def cumulative_frontiers_for_frames(records, frames):
    """Build successive frontiers while processing each record only once."""
    frontiers = []
    frontier = []
    next_record = 0

    for iteration in frames:
        new_records = []
        while (
            next_record < len(records)
            and records[next_record]["iteration"] <= iteration
        ):
            new_records.append(records[next_record])
            next_record += 1

        if new_records:
            # A dominated point can never return to a cumulative frontier, so
            # only the previous frontier and newly arrived records are needed.
            frontier = pareto_records(frontier + new_records)
        frontiers.append(frontier.copy())

    return frontiers


def render_movie(
    experiments,
    output_file,
    *,
    burn_in_epochs=5,
    frame_every_iterations=500,
    fps=5,
    linewidth=1.15,
    alpha=0.60,
    recon_min=None,
    recon_max=None,
    kl_min=None,
    kl_max=None,
):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Movie rendering requires imageio; install the project requirements"
        ) from exc

    if frame_every_iterations <= 0:
        raise ValueError("frame_every_iterations must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    limits = resolve_axis_limits(
        [experiment["retained_records"] for experiment in experiments],
        recon_min=recon_min,
        recon_max=recon_max,
        kl_min=kl_min,
        kl_max=kl_max,
    )
    frames = frame_iterations(experiments, frame_every_iterations)
    timeline = max(
        experiments,
        key=lambda experiment: experiment["retained_records"][-1]["iteration"],
    )["retained_records"]
    timeline_start = timeline[0]["iteration"]
    evolving_frontiers = [
        cumulative_frontiers_for_frames(experiment["retained_records"], frames)
        for experiment in experiments
    ]
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output_file, fps=fps) as writer:
        for frame_index, iteration in enumerate(frames):
            timeline_index = min(max(iteration - timeline_start, 0), len(timeline) - 1)
            epoch = timeline[timeline_index]["epoch"]
            step = timeline[timeline_index]["step"]
            frame_experiments = []
            for experiment, experiment_frontiers in zip(
                experiments, evolving_frontiers
            ):
                frame_experiment = dict(experiment)
                frame_experiment["frontier_records"] = experiment_frontiers[frame_index]
                frame_experiments.append(frame_experiment)

            groups = build_groups(frame_experiments)
            fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
            handles = plot_grouped_frontiers(
                ax, groups, linewidth=linewidth, alpha=alpha
            )
            configure_pareto_axes(
                ax,
                title=f"Pareto Frontier Evolution — Epoch {epoch}, Step {step}",
                **limits,
            )
            ax.legend(handles=handles, title="β / learning rate")

            canvas = FigureCanvas(fig)
            canvas.draw()
            frame = np.asarray(canvas.buffer_rgba())[:, :, :3]
            writer.append_data(frame)
            plt.close(fig)

    print(f"Saved {output_file}")
    return output_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a cumulative Pareto-frontier evolution movie."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--expts", type=int, nargs="+")
    selection.add_argument("--first-expt", type=int)
    parser.add_argument("--last-expt", type=int, default=None)
    parser.add_argument("--parent-dir", type=Path, default=Path("expts"))
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--burn-in-epochs", type=int, default=5)
    parser.add_argument("--frame-every-iterations", type=int, default=500)
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--linewidth", type=float, default=1.15)
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--recon-min", type=float, default=None)
    parser.add_argument("--recon-max", type=float, default=None)
    parser.add_argument("--kl-min", type=float, default=None)
    parser.add_argument("--kl-max", type=float, default=None)
    return parser.parse_args()


def selected_experiments(args):
    if args.expts is not None:
        if args.last_expt is not None:
            raise ValueError("--last-expt may only be used with --first-expt")
        return args.expts
    if args.last_expt is None:
        raise ValueError("--last-expt is required with --first-expt")
    if args.first_expt > args.last_expt:
        raise ValueError("--first-expt must be <= --last-expt")
    return list(range(args.first_expt, args.last_expt + 1))


def main():
    args = parse_args()
    if args.burn_in_epochs < 0:
        raise ValueError("--burn-in-epochs must be >= 0")
    if args.linewidth <= 0:
        raise ValueError("--linewidth must be > 0")
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must satisfy 0 < value <= 1")

    manual_limits = {
        "recon_min": args.recon_min,
        "recon_max": args.recon_max,
        "kl_min": args.kl_min,
        "kl_max": args.kl_max,
    }
    validate_axis_limits(**manual_limits)

    experiments = []
    for expt_num in selected_experiments(args):
        print(f"Loading expt_{expt_num}...")
        try:
            experiments.append(
                load_experiment(expt_num, args.parent_dir, args.burn_in_epochs)
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"Warning: skipping expt_{expt_num}: {exc}")

    if not experiments:
        raise RuntimeError("No usable experiments were found")

    if args.output_file is None:
        numbers = [experiment["expt_num"] for experiment in experiments]
        stem = "_".join(str(number) for number in numbers)
        args.output_file = args.parent_dir / "pareto_comps" / f"ParetoEvolution_{stem}.mp4"

    render_movie(
        experiments,
        args.output_file,
        burn_in_epochs=args.burn_in_epochs,
        frame_every_iterations=args.frame_every_iterations,
        fps=args.fps,
        linewidth=args.linewidth,
        alpha=args.alpha,
        **manual_limits,
    )


if __name__ == "__main__":
    main()
