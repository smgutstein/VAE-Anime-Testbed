#!/usr/bin/env python3
"""
Find saved VAE snapshot frames closest to the intersection of two experiments'
Pareto curves.

Method
------
1. Read each experiment's loss history.
2. Apply the same optional chronological skip used by
   VAE_Pareto_Comparisons.py.
3. Construct each exact two-objective Pareto frontier using recon_loss and
   kl_loss (both minimized).
4. Represent each frontier as a polyline in:

       x = recon_loss
       y = log10(kl_loss)

5. Normalize both axes jointly across the two curves. Exact intersections are
   unchanged by this scaling; normalization only matters when the sampled
   curves do not cross exactly and a closest-approach calculation is needed.
6. Find exact intersections between Pareto polyline segments.
   - If there are multiple intersections, choose the one best represented by
     available saved snapshots from both experiments.
   - If there is no exact intersection, find the closest points on the two
     polylines and use their midpoint as the comparison target.
7. For each experiment, choose the saved raw_images frame whose Pareto point
   is nearest to the target in the same normalized
   (recon_loss, log10(kl_loss)) space.

The intersection is therefore determined from the full Pareto curves first;
snapshot cadence is used only to select the frames to inspect.
"""

import argparse
import math
import re
from pathlib import Path

from utils import get_experiment_dir


SNAPSHOT_RE = re.compile(
    r"^image_at_epoch_(?P<epoch>\d+)_step(?P<step>\d+)\.png$"
)

EPS = 1e-12


def read_loss_records(expt_dir: Path) -> list[dict]:
    loss_file = expt_dir / "stats" / "losses_file.txt"
    if not loss_file.is_file():
        raise FileNotFoundError(f"Loss file does not exist: {loss_file}")

    records = []

    with loss_file.open("r", encoding="utf-8") as fh:
        next(fh, None)

        for line_number, line in enumerate(fh, start=2):
            fields = [field.strip() for field in line.split("--")]
            if len(fields) < 4:
                continue

            try:
                epoch = int(fields[0])
                step = int(fields[1])
                recon_loss = float(fields[2])
                kl_loss = float(fields[3])
            except ValueError:
                print(
                    f"Warning: ignoring malformed loss record at "
                    f"{loss_file}:{line_number}"
                )
                continue

            if not (
                math.isfinite(recon_loss)
                and math.isfinite(kl_loss)
                and kl_loss > 0.0
            ):
                continue

            records.append({
                "iteration": len(records),
                "epoch": epoch,
                "step": step,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
            })

    if not records:
        raise ValueError(f"No valid positive-KL loss records found in {loss_file}")

    return records


def pareto_records(records: list[dict]) -> list[dict]:
    """
    Exact two-objective minimization frontier, matching the logic used by
    VAE_Pareto_Comparisons.py.
    """
    ordered = sorted(
        records,
        key=lambda record: (record["recon_loss"], record["kl_loss"]),
    )

    collapsed = []
    for record in ordered:
        if collapsed and record["recon_loss"] == collapsed[-1]["recon_loss"]:
            if record["kl_loss"] < collapsed[-1]["kl_loss"]:
                collapsed[-1] = record
        else:
            collapsed.append(record)

    frontier = []
    best_kl = float("inf")

    for record in collapsed:
        if record["kl_loss"] < best_kl:
            frontier.append(record)
            best_kl = record["kl_loss"]

    if len(frontier) < 2:
        raise ValueError("Pareto frontier must contain at least two points")

    return frontier


def retained_pareto_records(
    expt_dir: Path,
    skip_fraction: float,
) -> list[dict]:
    records = read_loss_records(expt_dir)
    skip_points = int(skip_fraction * len(records))
    retained = records[skip_points:]

    if len(retained) < 2:
        raise ValueError(
            f"{expt_dir.name}: too few records remain after "
            f"skip_fraction={skip_fraction}"
        )

    return pareto_records(retained)


def find_snapshots(expt_dir: Path) -> dict[tuple[int, int], Path]:
    raw_image_dir = expt_dir / "raw_images"
    if not raw_image_dir.is_dir():
        raise FileNotFoundError(
            f"Raw image directory does not exist: {raw_image_dir}"
        )

    snapshots = {}

    for path in raw_image_dir.iterdir():
        if not path.is_file():
            continue

        match = SNAPSHOT_RE.match(path.name)
        if match is None:
            continue

        key = (int(match.group("epoch")), int(match.group("step")))
        snapshots[key] = path

    if not snapshots:
        raise FileNotFoundError(
            f"No snapshot images matching image_at_epoch_####_step####.png "
            f"found in {raw_image_dir}"
        )

    return snapshots


def pareto_snapshots(
    expt_dir: Path,
    frontier: list[dict],
) -> list[dict]:
    snapshots = find_snapshots(expt_dir)

    eligible = []
    for record in frontier:
        image_path = snapshots.get((record["epoch"], record["step"]))
        if image_path is not None:
            eligible.append({
                **record,
                "image_path": image_path,
            })

    if not eligible:
        raise ValueError(
            f"{expt_dir.name} has no Pareto points with corresponding "
            f"saved raw_images snapshots"
        )

    return eligible


def transformed_point(record: dict) -> tuple[float, float]:
    return record["recon_loss"], math.log10(record["kl_loss"])


def make_normalizer(
    frontier_a: list[dict],
    frontier_b: list[dict],
):
    """
    Joint min-max normalization over recon_loss and log10(kl_loss).
    """
    raw_points = [
        transformed_point(record)
        for record in frontier_a + frontier_b
    ]

    xs = [point[0] for point in raw_points]
    ys = [point[1] for point in raw_points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_scale = x_max - x_min
    y_scale = y_max - y_min

    if x_scale <= EPS:
        x_scale = 1.0
    if y_scale <= EPS:
        y_scale = 1.0

    def normalize_xy(x: float, y: float) -> tuple[float, float]:
        return (
            (x - x_min) / x_scale,
            (y - y_min) / y_scale,
        )

    def denormalize_xy(x: float, y: float) -> tuple[float, float]:
        return (
            x_min + x * x_scale,
            y_min + y * y_scale,
        )

    return normalize_xy, denormalize_xy


def normalized_curve(
    frontier: list[dict],
    normalize_xy,
) -> list[tuple[float, float]]:
    return [
        normalize_xy(*transformed_point(record))
        for record in frontier
    ]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def subtract(a, b):
    return a[0] - b[0], a[1] - b[1]


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def multiply(point, scalar):
    return point[0] * scalar, point[1] * scalar


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def distance_sq(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def segment_intersection(p, p2, q, q2):
    """
    Return an intersection point for two non-collinear closed segments,
    otherwise None.
    """
    r = subtract(p2, p)
    s = subtract(q2, q)
    r_cross_s = cross(r, s)
    q_minus_p = subtract(q, p)

    if abs(r_cross_s) <= EPS:
        return None

    t = cross(q_minus_p, s) / r_cross_s
    u = cross(q_minus_p, r) / r_cross_s

    if -EPS <= t <= 1.0 + EPS and -EPS <= u <= 1.0 + EPS:
        t = min(1.0, max(0.0, t))
        return add(p, multiply(r, t))

    return None


def closest_point_on_segment(point, a, b):
    ab = subtract(b, a)
    denom = dot(ab, ab)

    if denom <= EPS:
        return a

    t = dot(subtract(point, a), ab) / denom
    t = min(1.0, max(0.0, t))
    return add(a, multiply(ab, t))


def closest_points_between_segments(p1, p2, q1, q2):
    intersection = segment_intersection(p1, p2, q1, q2)
    if intersection is not None:
        return intersection, intersection, 0.0

    candidates = []

    q_for_p1 = closest_point_on_segment(p1, q1, q2)
    candidates.append((p1, q_for_p1, distance_sq(p1, q_for_p1)))

    q_for_p2 = closest_point_on_segment(p2, q1, q2)
    candidates.append((p2, q_for_p2, distance_sq(p2, q_for_p2)))

    p_for_q1 = closest_point_on_segment(q1, p1, p2)
    candidates.append((p_for_q1, q1, distance_sq(p_for_q1, q1)))

    p_for_q2 = closest_point_on_segment(q2, p1, p2)
    candidates.append((p_for_q2, q2, distance_sq(p_for_q2, q2)))

    return min(candidates, key=lambda candidate: candidate[2])


def exact_curve_intersections(curve_a, curve_b):
    intersections = []

    for i in range(len(curve_a) - 1):
        for j in range(len(curve_b) - 1):
            point = segment_intersection(
                curve_a[i],
                curve_a[i + 1],
                curve_b[j],
                curve_b[j + 1],
            )

            if point is None:
                continue

            duplicate = any(
                distance_sq(point, existing["point"]) <= 1e-20
                for existing in intersections
            )
            if duplicate:
                continue

            intersections.append({
                "point": point,
                "segment_a": i,
                "segment_b": j,
            })

    return intersections


def closest_curve_approach(curve_a, curve_b):
    best = None

    for i in range(len(curve_a) - 1):
        for j in range(len(curve_b) - 1):
            point_a, point_b, dist_sq = closest_points_between_segments(
                curve_a[i],
                curve_a[i + 1],
                curve_b[j],
                curve_b[j + 1],
            )

            candidate = {
                "point_a": point_a,
                "point_b": point_b,
                "distance_sq": dist_sq,
                "segment_a": i,
                "segment_b": j,
            }

            if best is None or dist_sq < best["distance_sq"]:
                best = candidate

    return best


def nearest_snapshot_to_target(
    snapshots: list[dict],
    target,
    normalize_xy,
) -> tuple[dict, float]:
    best_record = None
    best_dist_sq = float("inf")

    for record in snapshots:
        point = normalize_xy(*transformed_point(record))
        dist_sq = distance_sq(point, target)

        if dist_sq < best_dist_sq:
            best_record = record
            best_dist_sq = dist_sq

    return best_record, math.sqrt(best_dist_sq)


def choose_intersection_for_snapshots(
    intersections,
    snapshots_a,
    snapshots_b,
    normalize_xy,
):
    """
    If multiple exact intersections exist, choose the one best represented by
    available snapshots from both experiments.
    """
    best = None

    for intersection in intersections:
        target = intersection["point"]

        frame_a, dist_a = nearest_snapshot_to_target(
            snapshots_a,
            target,
            normalize_xy,
        )
        frame_b, dist_b = nearest_snapshot_to_target(
            snapshots_b,
            target,
            normalize_xy,
        )

        candidate = {
            **intersection,
            "target": target,
            "frame_a": frame_a,
            "frame_b": frame_b,
            "frame_distance_a": dist_a,
            "frame_distance_b": dist_b,
            "combined_frame_distance": math.hypot(dist_a, dist_b),
        }

        if (
            best is None
            or candidate["combined_frame_distance"]
            < best["combined_frame_distance"]
        ):
            best = candidate

    return best


def midpoint(a, b):
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def resolve_experiment_dir(parent_dir: Path, expt_num: int) -> Path:
    expt_dir = get_experiment_dir(parent_dir, expt_num)
    if not expt_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment directory does not exist: {expt_dir}"
        )
    return expt_dir


def print_frame(expt_dir: Path, point: dict, target_distance: float) -> None:
    print(f"{expt_dir.name}")
    print(f"  epoch:       {point['epoch']}")
    print(f"  step:        {point['step']}")
    print(f"  iteration:   {point['iteration']}")
    print(f"  recon_loss:  {point['recon_loss']:.8g}")
    print(f"  kl_loss:     {point['kl_loss']:.8g}")
    print(f"  image:       {point['image_path']}")
    print(f"  target_dist: {target_distance:.8g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find saved raw_images frames closest to the intersection "
            "(or closest approach) of two experiments' Pareto curves."
        )
    )
    parser.add_argument(
        "--expts",
        type=int,
        nargs=2,
        required=True,
        metavar=("EXPT_A", "EXPT_B"),
        help="Exactly two experiment numbers, e.g. --expts 483 491",
    )
    parser.add_argument(
        "--parent_dir",
        type=Path,
        default=Path("expts"),
        help="Parent directory containing expt_N directories (default: expts)",
    )
    parser.add_argument(
        "--skip_fraction",
        type=float,
        default=0.10,
        help=(
            "Earliest fraction of chronological loss records to exclude before "
            "calculating each Pareto frontier (default: 0.10, matching "
            "VAE_Pareto_Comparisons.py)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.skip_fraction < 1.0:
        raise ValueError("--skip_fraction must satisfy 0 <= value < 1")

    expt_a = resolve_experiment_dir(args.parent_dir, args.expts[0])
    expt_b = resolve_experiment_dir(args.parent_dir, args.expts[1])

    frontier_a = retained_pareto_records(
        expt_a,
        skip_fraction=args.skip_fraction,
    )
    frontier_b = retained_pareto_records(
        expt_b,
        skip_fraction=args.skip_fraction,
    )

    snapshots_a = pareto_snapshots(expt_a, frontier_a)
    snapshots_b = pareto_snapshots(expt_b, frontier_b)

    normalize_xy, denormalize_xy = make_normalizer(frontier_a, frontier_b)

    curve_a = normalized_curve(frontier_a, normalize_xy)
    curve_b = normalized_curve(frontier_b, normalize_xy)

    intersections = exact_curve_intersections(curve_a, curve_b)

    if intersections:
        chosen = choose_intersection_for_snapshots(
            intersections,
            snapshots_a,
            snapshots_b,
            normalize_xy,
        )

        target = chosen["target"]
        frame_a = chosen["frame_a"]
        frame_b = chosen["frame_b"]
        frame_distance_a = chosen["frame_distance_a"]
        frame_distance_b = chosen["frame_distance_b"]

        raw_recon, raw_log_kl = denormalize_xy(*target)
        raw_kl = 10.0 ** raw_log_kl

        print()
        print("Pareto-curve intersection comparison")
        print("====================================")
        print(f"Exact polyline intersections found: {len(intersections)}")
        print(f"Selected intersection recon_loss: {raw_recon:.8g}")
        print(f"Selected intersection kl_loss:    {raw_kl:.8g}")
        print()

    else:
        closest = closest_curve_approach(curve_a, curve_b)
        target = midpoint(closest["point_a"], closest["point_b"])

        frame_a, frame_distance_a = nearest_snapshot_to_target(
            snapshots_a,
            target,
            normalize_xy,
        )
        frame_b, frame_distance_b = nearest_snapshot_to_target(
            snapshots_b,
            target,
            normalize_xy,
        )

        raw_target_recon, raw_target_log_kl = denormalize_xy(*target)
        raw_target_kl = 10.0 ** raw_target_log_kl

        raw_a_recon, raw_a_log_kl = denormalize_xy(*closest["point_a"])
        raw_b_recon, raw_b_log_kl = denormalize_xy(*closest["point_b"])

        print()
        print("Pareto-curve closest-approach comparison")
        print("========================================")
        print("Exact polyline intersections found: 0")
        print(
            "Normalized curve-to-curve distance: "
            f"{math.sqrt(closest['distance_sq']):.8g}"
        )
        print(
            f"Closest point on {expt_a.name}: "
            f"recon={raw_a_recon:.8g}, "
            f"kl={10.0 ** raw_a_log_kl:.8g}"
        )
        print(
            f"Closest point on {expt_b.name}: "
            f"recon={raw_b_recon:.8g}, "
            f"kl={10.0 ** raw_b_log_kl:.8g}"
        )
        print(f"Comparison target recon_loss: {raw_target_recon:.8g}")
        print(f"Comparison target kl_loss:    {raw_target_kl:.8g}")
        print()

    print("Closest saved Pareto frames")
    print("---------------------------")
    print_frame(expt_a, frame_a, frame_distance_a)
    print()
    print_frame(expt_b, frame_b, frame_distance_b)
    print()
    print(
        "Frame-to-frame recon-loss difference: "
        f"{abs(frame_a['recon_loss'] - frame_b['recon_loss']):.8g}"
    )
    print(
        "Frame-to-frame KL-loss difference: "
        f"{abs(frame_a['kl_loss'] - frame_b['kl_loss']):.8g}"
    )


if __name__ == "__main__":
    main()
