import json

import pytest

from VAE_Checkpoint_FID_KID import (
    checkpoint_key,
    effective_kl_dimensions,
    load_progress,
    save_progress,
    weight_checkpoint_path,
)


def record(epoch=50, step=32, iteration=1682):
    return {"epoch": epoch, "step": step, "iteration": iteration}


def test_checkpoint_paths_follow_training_names(tmp_path):
    # The evaluator discovers full-state directories but loads the companion
    # weight-only file.  Both must use the training module's exact zero-padded
    # epoch/step convention or a long scan will fail only after training.
    item = record()
    assert checkpoint_key(item) == "epoch_0050_step_0032"
    assert weight_checkpoint_path(tmp_path, item) == (
        tmp_path / "checkpoints" / "epoch_0050_step_0032.weights.h5"
    )


def test_effective_dimensions():
    # Uniform KL uses every dimension equally; fully concentrated KL uses one.
    assert effective_kl_dimensions([2.0, 2.0, 2.0]) == pytest.approx(3.0)
    assert effective_kl_dimensions([4.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_progress_round_trip_and_signature_guard(tmp_path):
    # Resume should preserve a completed metric row only when the complete
    # evaluation signature still matches.  A changed image count stands in for
    # any changed reference/model/metric setting here.
    path = tmp_path / "Checkpoint_FID_KID.json"
    signature = {"reference": "abc", "n_images": 2000}
    item = {
        **record(),
        "checkpoint_key": "epoch_0050_step_0032",
        "metrics": {"generated_vs_raw": {}},
    }
    save_progress(
        path,
        experiment="expt_1",
        signature=signature,
        results_by_key={item["checkpoint_key"]: item},
        status="in_progress",
    )

    loaded = load_progress(path, signature, overwrite=False)
    assert loaded[item["checkpoint_key"]] == item
    assert json.loads(path.read_text())["status"] == "in_progress"

    with pytest.raises(ValueError, match="different evaluation settings"):
        load_progress(path, {**signature, "n_images": 1000}, overwrite=False)
    assert load_progress(path, {}, overwrite=True) == {}
