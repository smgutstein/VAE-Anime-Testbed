import numpy as np

from test_helpers import make_expt_dir, write_minimal_config


class TestAnalyzeResults:
    def test_analysis_runs_on_new_artifacts_only(self, tmp_path):
        from VAE_Anime_ArtifactWriter import ArtifactWriter
        from VAE_Anime_Analysis import AnalyzeResults

        parent_dir = tmp_path / "expts"
        parent_dir.mkdir()

        config_path = tmp_path / "config.ini"
        write_minimal_config(config_path, parent_dir=parent_dir, use_legacy_kl_names=False)

        expt_num = 7
        _, stats_dir, _, _, _ = make_expt_dir(parent_dir, expt_num)

        writer = ArtifactWriter(stats_dir).open()
        writer.write_loss_text_header()
        writer.write_loss_chunk(
            recon_loss=[10.0, 9.0, 8.0, 7.0],
            kl_loss=[0.1, 0.15, 0.2, 0.25],
            kl_weight=[1e-6, 2e-6, 3e-6, 4e-6],
            recon_ssim=[0.3, 0.4, 0.5, 0.6],
            active_latent_dims=[1, 2, 2, 3],
        )
        writer.write_latent_chunk(
            mu=[np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5])],
            log_var=[np.array([-1.0, -2.0, -3.0]), np.array([-1.5, -2.5, -3.5])],
        )
        writer.write_latent_var_chunk(
            mu_var=[np.array([0.1, 0.2, 0.3]), np.array([0.15, 0.25, 0.35])],
            log_var_var=[np.array([0.01, 0.02, 0.03]), np.array([0.015, 0.025, 0.035])],
        )
        writer.write_latent_kl_chunk(
            kl_per_dim=[
                np.array([0.01, 0.02, 0.03]),
                np.array([0.02, 0.03, 0.04]),
                np.array([0.03, 0.04, 0.05]),
                np.array([0.04, 0.05, 0.06]),
            ],
        )
        writer.close()

        ar = AnalyzeResults(config_file=str(config_path), expt=expt_num)

        recon, kl, weight = ar.get_recon_kl_results()
        assert recon == [10.0, 9.0, 8.0, 7.0]
        assert kl == [0.1, 0.15, 0.2, 0.25]
        assert weight == [1e-6, 2e-6, 3e-6, 4e-6]

        _, recon_pts, kl_pts = ar.read_loss_file_points()
        assert len(recon_pts) == 4
        assert len(kl_pts) == 4

        ar.make_singleton_graphs()

        assert (stats_dir / "Pareto_Comparisons.png").exists()
        assert (stats_dir / "Recon_KL_Comp_1.png").exists()
        assert (stats_dir / "Recon_KL_Comp_2.png").exists()
        assert (stats_dir / "Active_Latent_Dims.png").exists()
        assert (stats_dir / "KL_Per_Dim.png").exists()
        assert (stats_dir / "Recon_Loss_vs_SSIM.png").exists()
        assert (stats_dir / "Active_Dims_vs_KL_Loss.png").exists()
        assert (stats_dir / "Active_Dims_vs_Recon_Loss.png").exists()
        assert (stats_dir / "Active_Dims_Recon_KL_3D.html").exists()
        assert (stats_dir / "mu.png").exists()
        assert (stats_dir / "log_var.png").exists()

    def test_active_dims_relationship_movies_are_created(self, tmp_path, monkeypatch):
        import VAE_Anime_Analysis as analysis_module
        from VAE_Anime_ArtifactWriter import ArtifactWriter
        from VAE_Anime_Analysis import AnalyzeResults

        class FakeWriter:
            def __init__(self, outpath, fps):
                self.outpath = outpath
                self.fps = fps
                self.frames = 0

            def append_data(self, image):
                self.frames += 1

            def close(self):
                self.outpath.write_bytes(f"fake movie frames={self.frames}".encode("utf-8"))

        def fake_get_writer(outpath, fps):
            return FakeWriter(outpath, fps)

        monkeypatch.setattr(analysis_module.imageio, "get_writer", fake_get_writer)

        parent_dir = tmp_path / "expts"
        parent_dir.mkdir()

        config_path = tmp_path / "config.ini"
        write_minimal_config(config_path, parent_dir=parent_dir, use_legacy_kl_names=False)

        expt_num = 11
        _, stats_dir, _, movies_dir, _ = make_expt_dir(parent_dir, expt_num)

        writer = ArtifactWriter(stats_dir).open()
        writer.write_loss_text_header()
        writer.write_loss_chunk(
            recon_loss=[10.0, 9.0, 8.0, 7.0, 6.0],
            kl_loss=[0.1, 0.15, 0.2, 0.25, 0.3],
            kl_weight=[1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
            recon_ssim=[0.3, 0.4, 0.5, 0.6, 0.7],
            active_latent_dims=[1, 2, 2, 3, 4],
        )
        writer.close()

        ar = AnalyzeResults(config_file=str(config_path), expt=expt_num)
        result = ar.make_active_dims_relationship_movies()

        assert result == {
            "active_dims_vs_kl_loss": True,
            "active_dims_vs_recon_loss": True,
            "active_dims_loss_comparison": True,
        }
        assert (movies_dir / "active_dims_vs_kl_loss.mp4").exists()
        assert (movies_dir / "active_dims_vs_recon_loss.mp4").exists()
        assert (movies_dir / "active_dims_loss_comparison.mp4").exists()
