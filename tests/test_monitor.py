import numpy as np


def test_training_monitor_flushes_buffers_to_artifacts(tmp_path):
    from VAE_Anime_Training_Monitor import TrainingMonitor
    from VAE_Anime_ArtifactReader import ArtifactReader

    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()

    monitor = TrainingMonitor(stats_dir=stats_dir, snapshot_every=2)
    monitor.open()
    monitor.record_text_line(
        epoch=0,
        step=0,
        loss_recon=1.0,
        loss_kl=0.1,
        curr_kl_weight=0.01,
        test1=False,
        test2=False,
        num_maxes=0,
        max_kl_weight_seen=0.01,
        min_kl_weight_seen=0.01,
        window_len=1,
        kl_jump_ratio=1.0,
        max_log_var=0.0,
        min_log_var=0.0,
        max_grad_norm=5.0,
    )
    monitor.record_losses(
        1.0,
        0.1,
        0.01,
        recon_ssim=0.75,
        active_latent_dims=2,
    )
    monitor.record_latent_stats(
        mu_mean=np.array([1.0, 2.0]),
        log_var_mean=np.array([-1.0, -2.0]),
        mu_var=np.array([0.1, 0.2]),
        log_var_var=np.array([0.01, 0.02]),
        kl_per_dim=np.array([0.001, 0.02]),
    )
    monitor.flush()
    monitor.close()

    reader = ArtifactReader(stats_dir)
    loss = reader.read_loss_series()
    latent = reader.read_latent_series()
    latent_var = reader.read_latent_var_series()
    latent_kl = reader.read_latent_kl_series()

    assert loss.recon_loss == [1.0]
    assert loss.kl_loss == [0.1]
    assert loss.kl_weight == [0.01]
    assert loss.recon_ssim == [0.75]
    assert loss.active_latent_dims == [2]
    assert len(latent.mu) == 1
    assert len(latent.log_var) == 1
    assert len(latent_var.mu_var) == 1
    assert len(latent_var.log_var_var) == 1
    assert len(latent_kl.kl_per_dim) == 1
