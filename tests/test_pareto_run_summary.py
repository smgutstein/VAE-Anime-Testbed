import json

from VAE_Anime_ParetoRunSummary import (
    build_completion_summary,
    find_pareto_records,
    summarize_experiment,
    write_summary,
)


def _record(iteration, epoch, step, recon, kl):
    return {
        "iteration": iteration,
        "epoch": epoch,
        "step": step,
        "recon_loss": recon,
        "kl_loss": kl,
    }


def test_find_pareto_records_preserves_time_and_adds_ranks():
    records = [
        _record(0, 0, 0, 10.0, 1.0),
        _record(1, 0, 1, 8.0, 2.0),
        _record(2, 0, 2, 9.0, 3.0),  # dominated
        _record(3, 1, 0, 7.0, 4.0),
    ]

    frontier = find_pareto_records(records)

    assert [point["iteration"] for point in frontier] == [3, 1, 0]
    assert [point["pareto_rank_by_recon"] for point in frontier] == [1, 2, 3]
    assert [point["chronological_rank"] for point in frontier] == [3, 2, 1]


def test_completion_summary_uses_chronological_pareto_order():
    frontier = find_pareto_records([
        _record(2, 0, 2, 8.0, 2.0),
        _record(5, 1, 2, 7.0, 3.0),
        _record(8, 2, 2, 6.0, 4.0),
        _record(11, 3, 2, 5.0, 5.0),
    ])

    completion = build_completion_summary(frontier, percentages=(50, 75, 100))

    assert completion["50"]["iteration"] == 5
    assert completion["75"]["iteration"] == 8
    assert completion["100"]["iteration"] == 11


def test_summarize_experiment_respects_min_epoch_and_writes_json(tmp_path):
    expt_dir = tmp_path / "expt_7"
    stats_dir = expt_dir / "stats"
    stats_dir.mkdir(parents=True)
    (stats_dir / "losses_file.txt").write_text(
        "epoch -- step -- recon_loss -- kl_loss -- kl_weight\n"
        "0 -- 0 -- 10.0 -- 1.0 -- 1.0\n"
        "1 -- 0 -- 8.0 -- 3.0 -- 1.0\n"
        "1 -- 1 -- 7.0 -- 4.0 -- 1.0\n"
        "2 -- 0 -- 6.0 -- 5.0 -- 1.0\n",
        encoding="utf-8",
    )

    summary = summarize_experiment(expt_dir, min_epoch=1)
    json_path, text_path = write_summary(expt_dir, summary)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert saved["min_epoch"] == 1
    assert saved["last_pareto_point"]["epoch"] == 2
    assert saved["minimum_recon_pareto_point"]["recon_loss"] == 6.0
    assert saved["minimum_kl_pareto_point"]["kl_loss"] == 3.0
    assert json_path == stats_dir / "pareto_run_summary.json"
    assert text_path == stats_dir / "pareto_run_summary.txt"
    assert "Pareto Run Summary: expt_7" in text
    assert "Last Pareto point:" in text
    assert "100%:" in text
    assert "All Pareto points ordered by reconstruction loss" in text
