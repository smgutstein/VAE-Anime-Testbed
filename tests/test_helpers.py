from pathlib import Path
import configparser


def write_minimal_config(path: Path, parent_dir: Path, use_legacy_kl_names: bool):
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

    cfg["Safety_Parameters"] = {
        "max_grad_norm": "5.0",
        "step_guard_kl_jump_ratio_threshold": "100.0",
        "step_guard_kl_abs_threshold": "1000000.0",
        "step_guard_max_log_var_threshold": "20.0",
        "tripwire_lr_backoff": "0.5",
        "tripwire_lr_floor": "1e-6",
        "max_consecutive_tripwires": "3",
    }

    cfg["Monitoring_Parameters"] = {
        "snapshot_every": "1",
        "train_preview_count": "4",
        "valid_preview_count": "4",
        "take_initial_snapshot": "False",
        "run_analysis": "False",
        "make_mu_log_var_movies": "False",
        "active_dim_kl_threshold": "0.01",
    }

    cfg["Reproducibility"] = {
        "seed": "123",
        "deterministic": "True",
    }

    with open(path, "w") as f:
        cfg.write(f)


def make_expt_dir(parent_dir: Path, expt_num: int):
    expt_dir = parent_dir / f"expt_{expt_num}"
    stats_dir = expt_dir / "stats"
    raw_image_dir = expt_dir / "raw_images"
    movies_dir = expt_dir / "movies"
    model_info_dir = expt_dir / "model_info"

    stats_dir.mkdir(parents=True, exist_ok=True)
    raw_image_dir.mkdir(parents=True, exist_ok=True)
    movies_dir.mkdir(parents=True, exist_ok=True)
    model_info_dir.mkdir(parents=True, exist_ok=True)
    return expt_dir, stats_dir, raw_image_dir, movies_dir, model_info_dir


def make_config_ini_text(
    *,
    loss_policy="fixed_beta",
    beta="0.5",
    initial_kl_weight=None,
    max_kl_weight=None,
    kl_weight_update_factor=None,
    running_window=None,
    kl_adj_factor=None,
    kl_adj_factor_max=None,
    kl_adj_update_factor=None,
    epochs="10",
    learning_rate="0.001",
    parent_dir="/tmp/vae_test_output",
    save_net="false",
    expt_name=None,
    data_dir="/tmp/vae_test_data",
    batch_size="32",
    image_size="64",
    val_split="0.2",
    shuffle_buffer="100",
    train_drop_remainder="true",
    latent_dim="16",
    base_filters="8",
    filter_factors="1, 2, 4",
    encode_dense_units="64",
    kernel_size="3",
    max_grad_norm="5.0",
    step_guard_kl_jump_ratio_threshold="100.0",
    step_guard_kl_abs_threshold="1000000.0",
    step_guard_max_log_var_threshold="20.0",
    tripwire_lr_backoff="0.5",
    tripwire_lr_floor="1e-6",
    max_consecutive_tripwires="3",
    snapshot_every="100",
    train_preview_count="4",
    valid_preview_count="4",
    take_initial_snapshot="false",
    run_analysis="false",
    make_mu_log_var_movies="false",
    active_dim_kl_threshold="0.01",
    seed="42",
    deterministic="false",
    include_reproducibility=True,
    include_training=True,
    include_output=True,
    include_data=True,
    include_model=True,
    include_monitoring=True,
    include_safety=True,
):
    parts = []

    if include_training:
        parts.append("[Training_Parameters]")
        parts.append(f"epochs = {epochs}")
        parts.append(f"learning_rate = {learning_rate}")
        parts.append(f"loss_policy = {loss_policy}")

        if loss_policy == "fixed_beta" and beta is not None:
            parts.append(f"beta = {beta}")

        if loss_policy == "adaptive_kl":
            for key, val in [
                ("initial_kl_weight", initial_kl_weight),
                ("max_kl_weight", max_kl_weight),
                ("kl_weight_update_factor", kl_weight_update_factor),
                ("running_window", running_window),
                ("kl_adj_factor", kl_adj_factor),
                ("kl_adj_factor_max", kl_adj_factor_max),
                ("kl_adj_update_factor", kl_adj_update_factor),
            ]:
                if val is not None:
                    parts.append(f"{key} = {val}")

    if include_output:
        parts.append("[Output_Parameters]")
        parts.append(f"parent_dir = {parent_dir}")
        parts.append(f"save_net = {save_net}")
        if expt_name is not None:
            parts.append(f"expt_name = {expt_name}")

    if include_data:
        parts.append("[Data_Parameters]")
        parts.append(f"data_dir = {data_dir}")
        parts.append(f"batch_size = {batch_size}")
        parts.append(f"image_size = {image_size}")
        parts.append(f"val_split = {val_split}")
        parts.append(f"shuffle_buffer = {shuffle_buffer}")
        parts.append(f"train_drop_remainder = {train_drop_remainder}")

    if include_model:
        parts.append("[Model_Parameters]")
        parts.append(f"latent_dim = {latent_dim}")
        parts.append(f"base_filters = {base_filters}")
        parts.append(f"filter_factors = {filter_factors}")
        parts.append(f"encode_dense_units = {encode_dense_units}")
        parts.append(f"kernel_size = {kernel_size}")

    if include_safety:
        parts.append("[Safety_Parameters]")
        parts.append(f"max_grad_norm = {max_grad_norm}")
        parts.append(
            f"step_guard_kl_jump_ratio_threshold = {step_guard_kl_jump_ratio_threshold}"
        )
        parts.append(f"step_guard_kl_abs_threshold = {step_guard_kl_abs_threshold}")
        parts.append(
            f"step_guard_max_log_var_threshold = {step_guard_max_log_var_threshold}"
        )
        parts.append(f"tripwire_lr_backoff = {tripwire_lr_backoff}")
        parts.append(f"tripwire_lr_floor = {tripwire_lr_floor}")
        parts.append(f"max_consecutive_tripwires = {max_consecutive_tripwires}")

    if include_monitoring:
        parts.append("[Monitoring_Parameters]")
        parts.append(f"snapshot_every = {snapshot_every}")
        parts.append(f"train_preview_count = {train_preview_count}")
        parts.append(f"valid_preview_count = {valid_preview_count}")
        parts.append(f"take_initial_snapshot = {take_initial_snapshot}")
        parts.append(f"run_analysis = {run_analysis}")
        parts.append(f"make_mu_log_var_movies = {make_mu_log_var_movies}")
        parts.append(f"active_dim_kl_threshold = {active_dim_kl_threshold}")

    if include_reproducibility:
        parts.append("[Reproducibility]")
        parts.append(f"seed = {seed}")
        parts.append(f"deterministic = {deterministic}")

    return "\n".join(parts)


def write_config_text(tmp_path: Path, text: str):
    cfg_file = tmp_path / "config.ini"
    cfg_file.write_text(text)
    return cfg_file
