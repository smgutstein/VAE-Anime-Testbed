import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class TestArtifactRoundtrip:
    def test_loss_series_roundtrip(self, tmp_path):
        from VAE_Anime_ArtifactWriter import ArtifactWriter
        from VAE_Anime_ArtifactReader import ArtifactReader

        stats_dir = tmp_path / "stats"
        stats_dir.mkdir()

        writer = ArtifactWriter(stats_dir).open()
        writer.write_loss_text_header()
        writer.write_loss_chunk(
            recon_loss=[10.0, 9.0, 8.0],
            kl_loss=[0.1, 0.2, 0.3],
            kl_weight=[1e-6, 2e-6, 3e-6],
        )
        writer.write_latent_chunk(
            mu=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            log_var=[np.array([-1.0, -2.0]), np.array([-3.0, -4.0])],
        )
        writer.write_latent_var_chunk(
            mu_var=[np.array([0.1, 0.2]), np.array([0.3, 0.4])],
            log_var_var=[np.array([0.01, 0.02]), np.array([0.03, 0.04])],
        )
        writer.close()

        reader = ArtifactReader(stats_dir)
        loss_series = reader.read_loss_series()
        assert loss_series.recon_loss == [10.0, 9.0, 8.0]
        assert loss_series.kl_loss == [0.1, 0.2, 0.3]
        assert loss_series.kl_weight == [1e-6, 2e-6, 3e-6]

        latent_series = reader.read_latent_series()
        assert len(latent_series.mu) == 2
        assert len(latent_series.log_var) == 2

        latent_var_series = reader.read_latent_var_series()
        assert len(latent_var_series.mu_var) == 2
        assert len(latent_var_series.log_var_var) == 2


class TestRunArtifacts:
    def test_write_run_summary_writes_expected_json(self, tmp_path):
        from VAE_Anime_RunArtifacts import write_run_summary

        cfg = SimpleNamespace(
            expt_name="smoke_test",
            config_file=Path("config.ini"),
            seed=123,
            deterministic=True,
            epochs=5,
            learning_rate=0.001,
            latent_dim=8,
            batch_size=4,
            image_size=64,
            save_net=False,
            beta=None,
            initial_kl_weight=1e-6,
            max_kl_weight=10.0,
            kl_weight_update_factor=1.2,
            running_window=5,
        )
        loss_policy = SimpleNamespace(policy_name="adaptive_kl")

        write_run_summary(
            output_dir=tmp_path,
            cfg=cfg,
            loss_policy=loss_policy,
            status="completed",
            start_time=100.0,
            end_time=104.5,
            final_recon=12.3,
            final_kl=0.4,
            final_kl_weight=0.01,
        )

        data = json.loads((tmp_path / "run_summary.json").read_text())
        assert data["status"] == "completed"
        assert data["loss_policy"] == "adaptive_kl"
        assert data["runtime_seconds"] == 4.5
        assert data["final_recon_loss"] == 12.3
        assert data["final_kl_loss"] == 0.4
