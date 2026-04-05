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
        )
        writer.write_latent_chunk(
            mu=[np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5])],
            log_var=[np.array([-1.0, -2.0, -3.0]), np.array([-1.5, -2.5, -3.5])],
        )
        writer.write_latent_var_chunk(
            mu_var=[np.array([0.1, 0.2, 0.3]), np.array([0.15, 0.25, 0.35])],
            log_var_var=[np.array([0.01, 0.02, 0.03]), np.array([0.015, 0.025, 0.035])],
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

        assert (stats_dir / "Paretoish.png").exists()
        assert (stats_dir / "Recon_KL_Comp_1.png").exists()
        assert (stats_dir / "Recon_KL_Comp_2.png").exists()
        assert (stats_dir / "mu.png").exists()
        assert (stats_dir / "log_var.png").exists()
