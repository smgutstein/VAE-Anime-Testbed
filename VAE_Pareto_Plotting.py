"""Shared record filtering and plotting helpers for Pareto visualizations."""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


def records_after_burn_in(records, burn_in_epochs=5):
    """Return records at or after ``burn_in_epochs`` (epochs are zero-based)."""
    if burn_in_epochs < 0:
        raise ValueError("burn_in_epochs must be >= 0")
    return [record for record in records if record["epoch"] >= burn_in_epochs]


def frontier_array(frontier_records):
    """Convert frontier records to an ``(n, 2)`` recon/KL array."""
    return np.asarray(
        [
            [record["recon_loss"], record["kl_loss"]]
            for record in frontier_records
        ],
        dtype=float,
    ).reshape(-1, 2)


def validate_axis_limits(recon_min=None, recon_max=None, kl_min=None, kl_max=None):
    if recon_min is not None and recon_max is not None and recon_min >= recon_max:
        raise ValueError("recon_min must be less than recon_max")
    if kl_min is not None and kl_min <= 0:
        raise ValueError("kl_min must be > 0 for a logarithmic KL axis")
    if kl_max is not None and kl_max <= 0:
        raise ValueError("kl_max must be > 0 for a logarithmic KL axis")
    if kl_min is not None and kl_max is not None and kl_min >= kl_max:
        raise ValueError("kl_min must be less than kl_max")


def apply_axis_limits(ax, recon_min=None, recon_max=None, kl_min=None, kl_max=None):
    """Apply optional display-only limits without changing frontier membership."""
    validate_axis_limits(recon_min, recon_max, kl_min, kl_max)
    if recon_min is not None or recon_max is not None:
        ax.set_xlim(left=recon_min, right=recon_max)
    if kl_min is not None or kl_max is not None:
        ax.set_ylim(bottom=kl_min, top=kl_max)


def data_axis_limits(records_by_experiment):
    """Return fixed limits spanning all supplied records, with small margins."""
    records = [record for experiment in records_by_experiment for record in experiment]
    if not records:
        raise ValueError("Cannot calculate axis limits without records")

    recon = np.asarray([record["recon_loss"] for record in records], dtype=float)
    kl = np.asarray([record["kl_loss"] for record in records], dtype=float)

    recon_span = float(np.ptp(recon))
    recon_pad = 0.025 * recon_span if recon_span else max(abs(float(recon[0])) * 0.025, 1e-9)

    log_kl = np.log10(kl)
    log_span = float(np.ptp(log_kl))
    log_pad = 0.025 * log_span if log_span else 0.025

    return {
        "recon_min": float(np.min(recon) - recon_pad),
        "recon_max": float(np.max(recon) + recon_pad),
        "kl_min": float(10.0 ** (np.min(log_kl) - log_pad)),
        "kl_max": float(10.0 ** (np.max(log_kl) + log_pad)),
    }


def resolve_axis_limits(records_by_experiment, **manual_limits):
    """Fill unspecified limits from the full post-burn-in comparison data."""
    limits = data_axis_limits(records_by_experiment)
    for name, value in manual_limits.items():
        if value is not None:
            limits[name] = value
    validate_axis_limits(**limits)
    return limits


def plot_frontier(ax, frontier_records, *, color=None, linewidth=1.15, alpha=0.60):
    curve = frontier_array(frontier_records)
    if len(curve) == 0:
        return None
    return ax.plot(
        curve[:, 0],
        curve[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
    )[0]


def plot_grouped_frontiers(ax, groups, *, linewidth=1.15, alpha=0.60):
    """Plot each experiment frontier and return one legend handle per group."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = []

    for group_index, (_, group) in enumerate(groups.items()):
        color = colors[group_index % len(colors)]
        for experiment in group["experiments"]:
            plot_frontier(
                ax,
                experiment.get("frontier_records", []),
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
        handles.append(
            Line2D([0], [0], color=color, linewidth=2.5, label=group["label"])
        )

    return handles


def configure_pareto_axes(
    ax,
    *,
    title="Pareto Curves",
    recon_min=None,
    recon_max=None,
    kl_min=None,
    kl_max=None,
):
    ax.set_xlabel("Recon Loss")
    ax.set_ylabel("KL Loss")
    ax.set_yscale("log")
    ax.set_title(title)
    apply_axis_limits(ax, recon_min, recon_max, kl_min, kl_max)
