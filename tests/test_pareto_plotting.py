import pytest

from VAE_Pareto_Evolution_Movie import (
    cumulative_frontiers_for_frames,
    frame_iterations,
    frontier_through_iteration,
)
from VAE_Pareto_Plotting import (
    records_after_burn_in,
    resolve_axis_limits,
    validate_axis_limits,
)


def record(iteration, epoch, recon, kl):
    return {
        "iteration": iteration,
        "epoch": epoch,
        "recon_loss": recon,
        "kl_loss": kl,
    }


def test_burn_in_uses_epoch_not_run_fraction():
    records = [
        record(0, 0, 10.0, 10.0),
        record(1, 4, 9.0, 9.0),
        record(2, 5, 8.0, 8.0),
    ]
    assert records_after_burn_in(records, 5) == [records[2]]


def test_cumulative_frontier_changes_only_when_dominated():
    records = [
        record(0, 5, 5.0, 5.0),
        record(1, 5, 4.0, 6.0),
        record(2, 5, 4.0, 4.0),
    ]
    assert frontier_through_iteration(records, 1) == [records[1], records[0]]
    assert frontier_through_iteration(records, 2) == [records[2]]


def test_frame_iterations_include_final_iteration():
    experiments = [
        {"retained_records": [record(10, 5, 5.0, 5.0), record(26, 6, 4.0, 4.0)]},
        {"retained_records": [record(12, 5, 6.0, 6.0), record(20, 6, 5.0, 5.0)]},
    ]
    assert frame_iterations(experiments, 10) == [10, 20, 26]


def test_incremental_frontiers_match_direct_cumulative_calculation():
    records = [
        record(0, 5, 5.0, 5.0),
        record(1, 5, 4.0, 6.0),
        record(2, 5, 4.0, 4.0),
        record(3, 5, 3.0, 7.0),
    ]
    frames = [0, 2, 3]
    incremental = cumulative_frontiers_for_frames(records, frames)
    direct = [frontier_through_iteration(records, frame) for frame in frames]
    assert incremental == direct


def test_manual_limits_override_only_requested_bounds():
    records = [[record(0, 5, 10.0, 1e-2), record(1, 5, 20.0, 1e-1)]]
    limits = resolve_axis_limits(records, recon_min=7.0)
    assert limits["recon_min"] == 7.0
    assert limits["recon_max"] > 20.0
    assert limits["kl_min"] < 1e-2
    assert limits["kl_max"] > 1e-1


def test_invalid_log_axis_limit_is_rejected():
    with pytest.raises(ValueError, match="kl_min"):
        validate_axis_limits(kl_min=0.0)
