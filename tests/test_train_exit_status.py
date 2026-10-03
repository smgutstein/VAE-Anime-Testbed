"""Exit status of VAE_Anime_Train.main() when training diverges.

run_expt.sh relies on a diverged run (trip-wire limit or non-finite loss)
exiting with EXIT_TRAINING_DIVERGED so the batch continues with the next
config, while any other exception still propagates and stops the batch.
"""
import sys
from types import SimpleNamespace

import pytest


def _fake_trainer_class(exc):
    class FakeTrainer:
        def __init__(self, config_file, **kwargs):
            self.is_resuming = False
            self.curr_expt = 7
            self.output_dir = "expts/expt_7"
            self.cfg = SimpleNamespace(
                train_preview_count=0,
                valid_preview_count=0,
                take_initial_snapshot=False,
                run_analysis=True,
            )
            self.vae = SimpleNamespace(show_model=lambda: None)
            self.data = SimpleNamespace(display_sample_data=lambda *a: None)

        def train_loop(self):
            raise exc

    return FakeTrainer


def _patch_main(monkeypatch, train_module, exc):
    monkeypatch.setattr(train_module, "VAE_Trainer", _fake_trainer_class(exc))
    monkeypatch.setattr(train_module, "setup_logging", lambda level: None)

    def analysis_must_not_run(*args, **kwargs):
        raise AssertionError("analysis should not run after a failed train_loop")

    monkeypatch.setattr(train_module, "AnalyzeResults", analysis_must_not_run)
    monkeypatch.setattr(sys, "argv", ["VAE_Anime_Train.py", "-c", "cfg.ini"])


def test_diverged_training_returns_distinct_exit_status(monkeypatch, tf):
    import VAE_Anime_Train as train_module
    from VAE_Anime_StepGuard import TrainingDivergedError

    _patch_main(
        monkeypatch,
        train_module,
        TrainingDivergedError("Stopping after 3 consecutive trip-wires"),
    )

    assert train_module.main() == train_module.EXIT_TRAINING_DIVERGED
    assert train_module.EXIT_TRAINING_DIVERGED not in (0, 1, 2)


def test_other_training_errors_still_propagate(monkeypatch, tf):
    import VAE_Anime_Train as train_module

    _patch_main(monkeypatch, train_module, ValueError("real bug"))

    with pytest.raises(ValueError, match="real bug"):
        train_module.main()


def test_exit_status_matches_run_expt_script(tf):
    from pathlib import Path
    import VAE_Anime_Train as train_module

    script = (Path(__file__).resolve().parents[1] / "run_expt.sh").read_text()
    assert (
        f"EXIT_TRAINING_DIVERGED={train_module.EXIT_TRAINING_DIVERGED}\n" in script
    )
