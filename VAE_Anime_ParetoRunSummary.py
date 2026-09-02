"""Create per-experiment summaries of when final Pareto-front points appeared."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from utils import get_experiment_dir
from VAE_Loss_Records import (
    read_loss_records as shared_read_loss_records,
    pareto_records as shared_pareto_records,
)


DEFAULT_COMPLETION_PERCENTAGES = (50, 75, 80, 90, 95, 99, 100)
LOSS_FILE_NAME = "losses_file.txt"
JSON_OUTPUT_FILE_NAME = "pareto_run_summary.json"
TEXT_OUTPUT_FILE_NAME = "pareto_run_summary.txt"


def read_loss_records(loss_file: Path) -> list[dict]:
    """Read finite loss records while retaining epoch, step, and iteration."""
    return shared_read_loss_records(loss_file)


def find_pareto_records(records: Iterable[dict]) -> list[dict]:
    """Return the exact frontier ordered by recon loss, with rank fields."""
    frontier = [dict(record) for record in shared_pareto_records(records)]

    for recon_rank, record in enumerate(frontier, start=1):
        record["pareto_rank_by_recon"] = recon_rank

    chronological = sorted(frontier, key=lambda record: record["iteration"])
    chronological_rank = {
        record["iteration"]: rank
        for rank, record in enumerate(chronological, start=1)
    }
    for record in frontier:
        record["chronological_rank"] = chronological_rank[record["iteration"]]

    return frontier


def _point_copy(record: dict) -> dict:
    return {
        "epoch": record["epoch"],
        "step": record["step"],
        "iteration": record["iteration"],
        "recon_loss": record["recon_loss"],
        "kl_loss": record["kl_loss"],
    }


def build_completion_summary(
    pareto_records: list[dict],
    percentages: Iterable[int] = DEFAULT_COMPLETION_PERCENTAGES,
) -> dict:
    """Report when each requested share of the final frontier had appeared."""
    chronological = sorted(pareto_records, key=lambda record: record["iteration"])
    total = len(chronological)
    result = {}

    for percentage in percentages:
        if percentage <= 0 or percentage > 100:
            raise ValueError("Completion percentages must be in the interval (0, 100]")

        count = math.ceil(total * percentage / 100.0)
        record = chronological[count - 1]
        result[str(percentage)] = {
            "pareto_points_seen": count,
            "total_pareto_points": total,
            **_point_copy(record),
        }

    return result


def find_largest_pareto_gap(pareto_records: list[dict]) -> dict | None:
    """Return the largest iteration gap between consecutive Pareto discoveries."""
    chronological = sorted(pareto_records, key=lambda record: record["iteration"])
    if len(chronological) < 2:
        return None

    before, after = max(
        zip(chronological, chronological[1:]),
        key=lambda pair: pair[1]["iteration"] - pair[0]["iteration"],
    )
    return {
        "iterations": after["iteration"] - before["iteration"],
        "from_point": _point_copy(before),
        "to_point": _point_copy(after),
    }


def summarize_experiment(
    expt_dir: Path,
    min_epoch: int = 0,
    percentages: Iterable[int] = DEFAULT_COMPLETION_PERCENTAGES,
) -> dict:
    """Build a stopping-oriented Pareto summary for one completed experiment."""
    expt_dir = Path(expt_dir)
    loss_file = expt_dir / "stats" / LOSS_FILE_NAME
    all_records = read_loss_records(loss_file)
    eligible_records = [record for record in all_records if record["epoch"] >= min_epoch]

    if not eligible_records:
        raise ValueError(
            f"No finite loss records at or after epoch {min_epoch} in {loss_file}"
        )

    pareto_records = find_pareto_records(eligible_records)
    chronological = sorted(pareto_records, key=lambda record: record["iteration"])
    last_pareto = chronological[-1]
    final_record = max(all_records, key=lambda record: record["iteration"])

    return {
        "experiment": expt_dir.name,
        "source_file": str(loss_file.relative_to(expt_dir)),
        "min_epoch": min_epoch,
        "total_training_points": len(all_records),
        "eligible_training_points": len(eligible_records),
        "total_pareto_points": len(pareto_records),
        "final_training_point": _point_copy(final_record),
        "last_pareto_point": _point_copy(last_pareto),
        "last_pareto_training_fraction": (
            (last_pareto["iteration"] + 1) / (final_record["iteration"] + 1)
        ),
        "minimum_recon_pareto_point": _point_copy(
            min(pareto_records, key=lambda record: record["recon_loss"])
        ),
        "minimum_kl_pareto_point": _point_copy(
            min(pareto_records, key=lambda record: record["kl_loss"])
        ),
        "pareto_point_completion": build_completion_summary(
            pareto_records,
            percentages=percentages,
        ),
        "largest_pareto_gap": find_largest_pareto_gap(pareto_records),
        "pareto_points_by_recon_loss": pareto_records,
    }


def _format_point(point: dict) -> str:
    return (
        f"epoch={point['epoch']}, step={point['step']}, "
        f"iteration={point['iteration']}, "
        f"recon_loss={point['recon_loss']:.8g}, "
        f"kl_loss={point['kl_loss']:.8g}"
    )


def render_text_summary(summary: dict) -> str:
    """Render a human-readable version of a Pareto run summary."""
    lines = [
        f"Pareto Run Summary: {summary['experiment']}",
        "=" * (20 + len(summary["experiment"])),
        "",
        f"Source file: {summary['source_file']}",
        f"Minimum epoch included: {summary['min_epoch']}",
        f"Total training points: {summary['total_training_points']}",
        f"Eligible training points: {summary['eligible_training_points']}",
        f"Total Pareto points: {summary['total_pareto_points']}",
        (
            "Last Pareto training fraction: "
            f"{summary['last_pareto_training_fraction']:.2%}"
        ),
        "",
        "Key points",
        "----------",
        f"Final training point: {_format_point(summary['final_training_point'])}",
        f"Last Pareto point: {_format_point(summary['last_pareto_point'])}",
        (
            "Minimum reconstruction-loss Pareto point: "
            f"{_format_point(summary['minimum_recon_pareto_point'])}"
        ),
        (
            "Minimum KL-loss Pareto point: "
            f"{_format_point(summary['minimum_kl_pareto_point'])}"
        ),
        "",
        "Pareto-point completion",
        "-----------------------",
    ]

    for percentage, point in summary["pareto_point_completion"].items():
        lines.append(
            f"{percentage:>3}%: {point['pareto_points_seen']}/"
            f"{point['total_pareto_points']} points by "
            f"epoch={point['epoch']}, step={point['step']}, "
            f"iteration={point['iteration']}"
        )

    lines.extend([
        "",
        "Largest gap between Pareto discoveries",
        "--------------------------------------",
    ])
    gap = summary["largest_pareto_gap"]
    if gap is None:
        lines.append("Not applicable: fewer than two Pareto points.")
    else:
        lines.extend([
            f"Gap: {gap['iterations']} iterations",
            f"From: {_format_point(gap['from_point'])}",
            f"To:   {_format_point(gap['to_point'])}",
        ])

    lines.extend([
        "",
        "All Pareto points ordered by reconstruction loss",
        "------------------------------------------------",
        (
            "recon_rank  chronological_rank  epoch  step  iteration  "
            "recon_loss  kl_loss"
        ),
    ])
    for point in summary["pareto_points_by_recon_loss"]:
        lines.append(
            f"{point['pareto_rank_by_recon']:>10}  "
            f"{point['chronological_rank']:>18}  "
            f"{point['epoch']:>5}  {point['step']:>4}  "
            f"{point['iteration']:>9}  "
            f"{point['recon_loss']:>10.8g}  {point['kl_loss']:>10.8g}"
        )

    return "\n".join(lines) + "\n"


def write_summary(expt_dir: Path, summary: dict) -> tuple[Path, Path]:
    stats_dir = Path(expt_dir) / "stats"
    json_path = stats_dir / JSON_OUTPUT_FILE_NAME
    text_path = stats_dir / TEXT_OUTPUT_FILE_NAME

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    text_path.write_text(render_text_summary(summary), encoding="utf-8")
    return json_path, text_path


def resolve_experiment_dirs(
    parent_dir: Path,
    expt_nums: Iterable[int] | None,
    expt_dirs: Iterable[str] | None,
) -> list[Path]:
    resolved = []
    for expt_num in expt_nums or []:
        resolved.append(get_experiment_dir(parent_dir, expt_num))
    for expt_dir in expt_dirs or []:
        resolved.append(Path(expt_dir))

    if not resolved:
        raise ValueError("Provide at least one experiment via --expts or --expt_dirs")

    missing = [path for path in resolved if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Experiment directory does not exist: {missing[0]}")

    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize when final Pareto-front points appeared during training."
    )
    parser.add_argument(
        "--parent_dir",
        type=Path,
        default=Path("expts"),
        help="Parent directory containing experiment directories (default: expts)",
    )
    parser.add_argument(
        "--expts",
        type=int,
        nargs="+",
        action="extend",
        default=None,
        help="Experiment numbers; the option may be repeated",
    )
    parser.add_argument(
        "--expt_dirs",
        nargs="+",
        action="extend",
        default=None,
        help="Explicit experiment directories; the option may be repeated",
    )
    parser.add_argument(
        "--min_epoch",
        type=int,
        default=0,
        help="Ignore loss records before this epoch (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_epoch < 0:
        raise ValueError("--min_epoch must be nonnegative")

    expt_dirs = resolve_experiment_dirs(
        parent_dir=args.parent_dir,
        expt_nums=args.expts,
        expt_dirs=args.expt_dirs,
    )

    for expt_dir in expt_dirs:
        summary = summarize_experiment(expt_dir, min_epoch=args.min_epoch)
        json_path, text_path = write_summary(expt_dir, summary)
        last_point = summary["last_pareto_point"]
        print(
            f"{expt_dir.name}: {summary['total_pareto_points']} Pareto points; "
            f"last at epoch {last_point['epoch']}, step {last_point['step']}, "
            f"iteration {last_point['iteration']}"
        )
        print(f"Saved {json_path}")
        print(f"Saved {text_path}")


if __name__ == "__main__":
    main()
