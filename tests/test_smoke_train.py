import json

from test_helpers import make_config_ini_text, write_config_text


def test_trainer_smoke_runs_one_tiny_epoch_and_writes_artifacts(tmp_path, monkeypatch, tf):
    from VAE_Anime_ArtifactReader import ArtifactReader
    from VAE_Anime_Train import VAE_Trainer
    from VAE_Anime_Datasets import Datasets
    from VAE_Anime_Snapshotter import VAESnapshotter

    parent_dir = tmp_path / "expts"
    data_dir = tmp_path / "data"

    cfg_file = write_config_text(
        tmp_path,
        make_config_ini_text(
            loss_policy="fixed_beta",
            beta="1.0",
            epochs="1",
            learning_rate="1e-4",
            parent_dir=str(parent_dir),
            data_dir=str(data_dir),
            batch_size="2",
            image_size="16",
            val_split="0.5",
            shuffle_buffer="4",
            train_drop_remainder="false",
            latent_dim="4",
            base_filters="8",
            filter_factors="1, 2, 4",
            encode_dense_units="32",
            kernel_size="3",
            snapshot_every="1000",
            train_preview_count="0",
            valid_preview_count="0",
            take_initial_snapshot="false",
            run_analysis="false",
            make_mu_log_var_movies="false",
            save_net="false",
            seed="123",
            deterministic="false",
        ),
    )

    def fake_download_data(self):
        self.data_downloaded = True

    def fake_make_train_and_validation_sets(self):
        x_train = tf.random.uniform((4, 16, 16, 3), dtype=tf.float32)
        x_valid = tf.random.uniform((2, 16, 16, 3), dtype=tf.float32)

        self.training_dataset = (
            tf.data.Dataset.from_tensor_slices(x_train)
            .batch(self.batch_size, drop_remainder=self.train_drop_remainder)
        )
        self.validation_dataset = (
            tf.data.Dataset.from_tensor_slices(x_valid)
            .batch(self.batch_size, drop_remainder=False)
        )
        self.datasets_made = True

    monkeypatch.setattr(Datasets, "download_data", fake_download_data)
    monkeypatch.setattr(Datasets, "make_train_and_validation_sets", fake_make_train_and_validation_sets)
    monkeypatch.setattr(
        Datasets,
        "training_dataset_for_epoch",
        lambda self, epoch: self.training_dataset,
    )
    monkeypatch.setattr(VAESnapshotter, "save_snapshot", lambda *args, **kwargs: None)

    trainer = VAE_Trainer(str(cfg_file))
    trainer.train_loop()

    assert trainer.output_dir.exists()
    assert trainer.stats_dir.exists()
    assert trainer.raw_image_dir.exists()
    assert trainer.movies_dir.exists()
    assert trainer.model_info_dir.exists()

    summary_file = trainer.output_dir / "run_summary.json"
    assert summary_file.exists()

    with open(summary_file, "r") as f:
        summary = json.load(f)

    assert summary["status"] == "completed"
    assert summary["loss_policy"] == "fixed_beta"
    assert summary["epochs"] == 1
    assert summary["batch_size"] == 2
    assert summary["image_size"] == 16
    assert summary["final_recon_loss"] is not None
    assert summary["final_kl_loss"] is not None
    assert summary["final_kl_weight"] is not None

    reader = ArtifactReader(trainer.stats_dir)
    loss_series = reader.read_loss_series()
    latent_series = reader.read_latent_series()
    latent_var_series = reader.read_latent_var_series()

    assert len(loss_series.recon_loss) > 0
    assert len(loss_series.kl_loss) > 0
    assert len(loss_series.kl_weight) > 0

    assert len(latent_series.mu) > 0
    assert len(latent_series.log_var) > 0

    assert len(latent_var_series.mu_var) > 0
    assert len(latent_var_series.log_var_var) > 0