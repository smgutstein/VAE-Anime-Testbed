from pathlib import Path
from types import SimpleNamespace


def test_experiment_run_create_makes_expected_layout(tmp_path, monkeypatch):
    import VAE_Anime_ExperimentRun as mod

    config_file = tmp_path / "config.ini"
    config_file.write_text("[dummy]\n")

    cfg = SimpleNamespace(
        parent_dir=tmp_path / "expts",
        config_file=config_file,
        loss_policy="adaptive_kl",
        seed=123,
        deterministic=True,
    )

    monkeypatch.setattr(mod, "get_git_hash", lambda: "fakehash")
    monkeypatch.setattr(
        mod,
        "snapshot_source_state",
        lambda output_dir: "fake source snapshot",
    )

    run = mod.ExperimentRun.create(cfg)

    assert run.expt_num == 1
    assert run.output_dir.exists()
    assert run.raw_image_dir.exists()
    assert run.stats_dir.exists()
    assert run.movies_dir.exists()
    assert run.model_info_dir.exists()
    assert (run.output_dir / "config.ini").exists()
    notes = (run.output_dir / "Notes.txt").read_text()
    assert "fakehash" in notes
    assert "fake source snapshot" in notes
    assert "Loss Policy: adaptive_kl" in notes
