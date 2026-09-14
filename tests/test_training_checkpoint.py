from pathlib import Path

from VAE_Anime_TrainingCheckpoint import (
    find_latest_training_checkpoint,
    preserve_and_rollback_artifacts,
    resolve_training_checkpoint,
)


def test_find_latest_training_checkpoint(tmp_path):
    root = tmp_path / "checkpoints" / "training_state"
    for name in ("epoch_0050_step_0032", "epoch_0100_step_0032"):
        path = root / name
        path.mkdir(parents=True)
        (path / "resume_state.json").write_text('{"schema_version": 1}')
        (path / "training_state.index").write_bytes(b"index")
        (path / "training_state.data-00000-of-00001").write_bytes(b"data")

    assert find_latest_training_checkpoint(tmp_path).name == "epoch_0100_step_0032"


def test_latest_checkpoint_ignores_incomplete_newer_directory(tmp_path):
    root = tmp_path / "checkpoints" / "training_state"
    complete = root / "epoch_0050_step_0032"
    complete.mkdir(parents=True)
    (complete / "resume_state.json").write_text('{"schema_version": 1}')
    (complete / "training_state.index").write_bytes(b"index")
    (complete / "training_state.data-00000-of-00001").write_bytes(b"data")
    incomplete = root / "epoch_0100_step_0032"
    incomplete.mkdir()
    (incomplete / "resume_state.json").write_text('{"schema_version": 1}')

    assert find_latest_training_checkpoint(tmp_path) == complete


def test_resolve_bare_checkpoint_name_in_training_state_dir(tmp_path):
    checkpoint = (
        tmp_path / "checkpoints" / "training_state" / "epoch_0050_step_0032"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "resume_state.json").write_text('{"schema_version": 1}')
    (checkpoint / "training_state.index").write_bytes(b"index")
    (checkpoint / "training_state.data-00000-of-00001").write_bytes(b"data")

    assert resolve_training_checkpoint(tmp_path, checkpoint.name) == checkpoint


def test_resolve_rejects_weight_only_checkpoint(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="not restart checkpoints"):
        resolve_training_checkpoint(tmp_path, "epoch_0050_step_0032.weights.h5")


def test_rollback_preserves_tail_before_truncating(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    artifact = stats / "losses_file.txt"
    artifact.write_bytes(b"kept-tail")

    backup = preserve_and_rollback_artifacts(
        tmp_path,
        {"losses_file.txt": 4},
        epoch=50,
    )

    assert artifact.read_bytes() == b"kept"
    assert (backup / "losses_file.txt.tail").read_bytes() == b"-tail"


    assert (backup / "stats" / "loss_events.pkl").is_file()
