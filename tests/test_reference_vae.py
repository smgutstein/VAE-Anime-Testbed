import json
from pathlib import Path
from types import SimpleNamespace

from VAE_Anime_ReferenceVAE import BestSSIMReferenceSaver


class _FakeModel:
    def __init__(self):
        self.saved = []
        self.weights_saved = []

    def save(self, path, overwrite=True):
        Path(path).write_text("model")
        self.saved.append(Path(path).name)

    def save_weights(self, path, overwrite=True):
        Path(path).write_text("weights")
        self.weights_saved.append(Path(path).name)


def _cfg(tmp_path):
    config_file = tmp_path / "source_config.ini"
    config_file.write_text("[dummy]\n")
    return SimpleNamespace(
        config_file=config_file,
        loss_policy="fixed_beta",
        beta=1.0,
        initial_kl_weight=None,
        max_kl_weight=None,
        kl_weight_update_factor=None,
        running_window=None,
        learning_rate=0.0002,
        seed=1776,
        deterministic=True,
        image_size=64,
        latent_dim=512,
        base_filters=32,
        filter_factors=(1, 2, 4),
        encode_dense_units=1024,
        kernel_size=3,
        data_dir=tmp_path / "data",
        batch_size=1500,
        val_split=0.2,
        shuffle_buffer=1000,
        train_drop_remainder=True,
    )


def _val(ssim):
    return SimpleNamespace(
        recon_mu=0.01,
        recon_sampled=0.02,
        kl_loss=0.3,
        ssim_mu=ssim,
        ssim_sampled=ssim - 0.01,
        active_latent_dims=20,
        agg_post_var_mean=1.0,
        agg_post_var_min=0.8,
        agg_post_var_max=1.2,
        n_images=100,
    )


def test_reference_saver_keeps_only_best_ssim(tmp_path, monkeypatch):
    import VAE_Anime_ReferenceVAE as mod

    # SimpleNamespace is not a dataclass, so provide the same serializable
    # validation payload the real ValidationResult dataclass would produce.
    monkeypatch.setattr(mod, "asdict", lambda value: vars(value))

    expts_dir = tmp_path / "expts"
    output_dir = expts_dir / "expt_1"
    output_dir.mkdir(parents=True)
    (output_dir / "Notes.txt").write_text("Git Hash: fake\n")

    vae = SimpleNamespace(
        vae_net=_FakeModel(),
        encoder=SimpleNamespace(encoder_net=_FakeModel()),
    )
    saver = BestSSIMReferenceSaver(output_dir, _cfg(tmp_path))

    assert saver.consider(vae=vae, val_result=_val(0.80), epoch=10, step=32, kl_weight=1.0)
    assert not saver.consider(vae=vae, val_result=_val(0.79), epoch=20, step=32, kl_weight=1.0)
    assert saver.consider(vae=vae, val_result=_val(0.85), epoch=30, step=32, kl_weight=1.0)

    ref_dir = tmp_path / "ref_vae" / "expt_1"
    assert (ref_dir / "reference_vae.weights.h5").exists()
    assert (ref_dir / "metadata.json").exists()
    assert (ref_dir / "config.ini").exists()
    assert (ref_dir / "Notes.txt").exists()

    metadata = json.loads((ref_dir / "metadata.json").read_text())
    assert metadata["selection_criterion"].startswith("maximum validation ssim_mu")
    assert metadata["selected_state"]["epoch"] == 30
    assert metadata["selected_state"]["validation_ssim_mu"] == 0.85
