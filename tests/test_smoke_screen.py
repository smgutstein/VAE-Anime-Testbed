from pathlib import Path
import configparser

import numpy as np

from VAE_Anime_Config import TrainerConfig
from VAE_Anime_ArtifactWriter import ArtifactWriter
from VAE_Anime_ArtifactReader import ArtifactReader
from VAE_Anime_Analysis import AnalyzeResults


def _write_minimal_config(
    path: Path,
    parent_dir: Path,
    use_legacy_kl_names: bool,
):
    cfg = configparser.ConfigParser()

    cfg["Training_Parameters"] = {
        "epochs": "2",
        "learning_rate": "0.001",
        "loss_policy": "adaptive_kl",
        "beta": "1.0",
        "running_window": "5",
    }

    if use_legacy_kl_names:
        cfg["Training_Parameters"]["kl_adj_factor"] = "1e-6"
        cfg["Training_Parameters"]["kl_adj_factor_max"] = "10.0"
        cfg["Training_Parameters"]["kl_adj_update_factor"] = "1.2"
    else:
        cfg["Training_Parameters"]["initial_kl_weight"] = "1e-6"
        cfg["Training_Parameters"]["max_kl_weight"] = "10.0"
        cfg["Training_Parameters"]["kl_weight_update_factor"] = "1.2"

    cfg["Output_Parameters"] = {
        "parent_dir": str(parent_dir),
        "save_net": "False",
        "expt_name": "smoke_test",
    }

    cfg["Data_Parameters"] = {
        "data_dir": str(parent_dir / "data"),
        "batch_size": "4",
        "image_size": "64",
        "val_split": "0.1",
        "shuffle_buffer": "100",
        "train_drop_remainder": "False",
    }

    cfg["Model_Parameters"] = {
        "latent_dim": "8",
        "base_filters": "32",
        "filter_factors": "1,2,4",
        "encode_dense_units": "128",
        "kernel_size": "3",
    }

    cfg["Monitoring_Parameters"] = {
        "snapshot_every": "1",
        "train_preview_count": "4",
        "valid_preview_count": "4",
        "take_initial_snapshot": "False",
        "run_analysis": "False",
        "make_mu_log_var_movies": "False",
    }

    cfg["Reproducibility"] = {
        "seed": "123",
        "deterministic": "True",
    }

    with open(path, "w") as f:
        cfg.write(f)


def _make_expt_dir(parent_dir: Path, expt_num: int):
    expt_dir = parent_dir / f"expt_{expt_num}"
    stats_dir = expt_dir / "stats"
    raw_image_dir = expt_dir / "raw_images"
    movies_dir = expt_dir / "movies"

    stats_dir.mkdir(parents=True, exist_ok=True)
    raw_image_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)
    return expt_dir, stats_dir, raw_image_dir, movies_dir


def test_config_loads_with_new_kl_names(tmp_path):
    config_path = tmp_path / "config_new.ini"
    _write_minimal_config(
        config_path,
        parent_dir=tmp_path / "expts",
        use_legacy_kl_names=False,
    )

    cfg = TrainerConfig.from_file(config_path)

    assert cfg.initial_kl_weight == 1e-6
    assert cfg.max_kl_weight == 10.0
    assert cfg.kl_weight_update_factor == 1.2


def test_config_loads_with_legacy_kl_names(tmp_path):
    config_path = tmp_path / "config_old.ini"
    _write_minimal_config(
        config_path,
        parent_dir=tmp_path / "expts",
        use_legacy_kl_names=True,
    )

    cfg = TrainerConfig.from_file(config_path)

    assert cfg.initial_kl_weight == 1e-6
    assert cfg.max_kl_weight == 10.0
    assert cfg.kl_weight_update_factor == 1.2


def test_artifact_writer_reader_roundtrip(tmp_path):
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


def test_analysis_runs_on_new_artifacts_only(tmp_path):
    parent_dir = tmp_path / "expts"
    parent_dir.mkdir()

    config_path = tmp_path / "config.ini"
    _write_minimal_config(
        config_path,
        parent_dir=parent_dir,
        use_legacy_kl_names=False,
    )

    expt_num = 7
    expt_dir, stats_dir, raw_image_dir, movies_dir = _make_expt_dir(parent_dir, expt_num)

    writer = ArtifactWriter(stats_dir).open()
    writer.write_loss_text_header()
    writer.write_loss_chunk(
        recon_loss=[10.0, 9.0, 8.0, 7.0],
        kl_loss=[0.1, 0.15, 0.2, 0.25],
        kl_weight=[1e-6, 2e-6, 3e-6, 4e-6],
    )
    writer.write_latent_chunk(
        mu=[
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 2.5, 3.5]),
        ],
        log_var=[
            np.array([-1.0, -2.0, -3.0]),
            np.array([-1.5, -2.5, -3.5]),
        ],
    )
    writer.write_latent_var_chunk(
        mu_var=[
            np.array([0.1, 0.2, 0.3]),
            np.array([0.15, 0.25, 0.35]),
        ],
        log_var_var=[
            np.array([0.01, 0.02, 0.03]),
            np.array([0.015, 0.025, 0.035]),
        ],
    )
    writer.close()

    assert not (stats_dir / "loss_lists.pkl").exists()
    assert not (stats_dir / "mu_log_var_lists.pkl").exists()
    assert not (stats_dir / "mu_log_var_lists2.pkl").exists()

    ar = AnalyzeResults(config_file=str(config_path), expt=expt_num)

    recon, kl, weight = ar.get_recon_kl_results()
    assert recon == [10.0, 9.0, 8.0, 7.0]
    assert kl == [0.1, 0.15, 0.2, 0.25]
    assert weight == [1e-6, 2e-6, 3e-6, 4e-6]

    _, recon_pts, kl_pts = ar.read_loss_file_points()
    assert len(recon_pts) == 4
    assert len(kl_pts) == 4

    ar.loss_plotter.compare_recon_kl_losses()
    ar.loss_plotter.compare_recon_kl_losses2()

    if hasattr(ar.latent_plotter, "make_final_mu_log_var_graphs"):
        ar.latent_plotter.make_final_mu_log_var_graphs()

    assert (stats_dir / "Recon_KL_Comp_1.png").exists()
    assert (stats_dir / "Recon_KL_Comp_2.png").exists()

    if hasattr(ar.latent_plotter, "make_final_mu_log_var_graphs"):
        assert (stats_dir / "mu.png").exists()
        assert (stats_dir / "log_var.png").exists()