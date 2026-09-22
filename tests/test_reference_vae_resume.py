import json

from VAE_Anime_ReferenceVAE import BestSSIMReferenceSaver


def _write_metadata(tmp_path, ssim):
    output_dir = tmp_path / "expts" / "expt_7"
    output_dir.mkdir(parents=True)
    ref_dir = tmp_path / "ref_vae" / "expt_7"
    ref_dir.mkdir(parents=True)
    (ref_dir / "metadata.json").write_text(
        json.dumps({"selected_state": {"validation_ssim_mu": ssim}}),
        encoding="utf-8",
    )
    return output_dir


def test_fresh_run_ignores_stale_reference_metadata(tmp_path):
    output_dir = _write_metadata(tmp_path, 0.9)
    saver = BestSSIMReferenceSaver(output_dir, cfg=None)
    assert saver.best_ssim == float("-inf")


def test_resumed_run_recovers_reference_best_ssim(tmp_path):
    output_dir = _write_metadata(tmp_path, 0.9)
    saver = BestSSIMReferenceSaver(output_dir, cfg=None, resume=True)
    assert saver.best_ssim == 0.9
