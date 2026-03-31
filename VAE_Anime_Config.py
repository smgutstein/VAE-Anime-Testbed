from dataclasses import dataclass
from pathlib import Path

from utils import (
    get_data_dir,
    get_parent_dir,
    get_seed_and_determinism,
    load_and_validate_config,
    parse_bool,
)


@dataclass(frozen=True)
class TrainerConfig:
    config_file: Path

    # Training
    epochs: int
    learning_rate: float
    loss_policy: str
    beta: float
    kl_adj_factor: float
    kl_adj_factor_max: float
    kl_adj_update_factor: float
    running_window: int

    # Output
    parent_dir: Path
    save_net: bool
    expt_name: str | None

    # Data
    data_dir: Path
    batch_size: int
    image_size: int
    val_split: float
    shuffle_buffer: int
    train_drop_remainder: bool

    # Model
    latent_dim: int
    base_filters: int
    filter_factors: tuple[int, int, int]
    encode_dense_units: int
    kernel_size: int

    # Monitoring
    snapshot_every: int
    train_preview_count: int
    valid_preview_count: int
    take_initial_snapshot: bool
    run_analysis: bool
    make_mu_log_var_movies: bool

    # Reproducibility
    seed: int
    deterministic: bool

    @classmethod
    def from_file(cls, config_file="config.ini"):
        config_path = Path(config_file)
        config = load_and_validate_config(config_path)

        filter_factors = tuple(
            int(x.strip())
            for x in config.get("Model_Parameters", "filter_factors").split(",")
        )
        if len(filter_factors) != 3:
            raise ValueError(
                "Model_Parameters.filter_factors must contain exactly 3 integers "
                "because the current encoder/decoder architecture has exactly 3 "
                "convolution stages."
            )

        seed, deterministic = get_seed_and_determinism(config)

        obj = cls(
            config_file=config_path,

            # Training
            epochs=config.getint("Training_Parameters", "epochs"),
            learning_rate=config.getfloat("Training_Parameters", "learning_rate"),
            loss_policy=config.get("Training_Parameters", "loss_policy"),
            beta=config.getfloat("Training_Parameters", "beta"),
            kl_adj_factor=config.getfloat("Training_Parameters", "kl_adj_factor"),
            kl_adj_factor_max=config.getfloat("Training_Parameters", "kl_adj_factor_max"),
            kl_adj_update_factor=config.getfloat("Training_Parameters", "kl_adj_update_factor"),
            running_window=config.getint("Training_Parameters", "running_window"),

            # Output
            parent_dir=get_parent_dir(config),
            save_net=parse_bool(config.get("Output_Parameters", "save_net"), "save_net"),
            expt_name=config.get("Output_Parameters","expt_name",fallback=None),

            # Data
            data_dir=get_data_dir(config),
            batch_size=config.getint("Data_Parameters", "batch_size"),
            image_size=config.getint("Data_Parameters", "image_size"),
            val_split=config.getfloat("Data_Parameters", "val_split"),
            shuffle_buffer=config.getint("Data_Parameters", "shuffle_buffer"),
            train_drop_remainder=parse_bool(
                config.get("Data_Parameters", "train_drop_remainder"),
                "train_drop_remainder",
            ),

            # Model
            latent_dim=config.getint("Model_Parameters", "latent_dim"),
            base_filters=config.getint("Model_Parameters", "base_filters"),
            filter_factors=filter_factors,
            encode_dense_units=config.getint("Model_Parameters", "encode_dense_units"),
            kernel_size=config.getint("Model_Parameters", "kernel_size"),

            # Monitoring
            snapshot_every=config.getint("Monitoring_Parameters", "snapshot_every"),
            train_preview_count=config.getint("Monitoring_Parameters", "train_preview_count"),
            valid_preview_count=config.getint("Monitoring_Parameters", "valid_preview_count"),
            take_initial_snapshot=parse_bool(
                config.get("Monitoring_Parameters", "take_initial_snapshot"),
                "take_initial_snapshot",
            ),
            run_analysis=parse_bool(
                config.get("Monitoring_Parameters", "run_analysis"),
                "run_analysis",
            ),
            make_mu_log_var_movies=parse_bool(
                config.get("Monitoring_Parameters", "make_mu_log_var_movies"),
                "make_mu_log_var_movies",
            ),

            # Reproducibility
            seed=seed,
            deterministic=deterministic,
        )

        obj.validate()
        return obj

    def validate(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.loss_policy not in {"adaptive_kl", "fixed_beta"}:
            raise ValueError("loss_policy must be one of: adaptive_kl, fixed_beta")
        if self.beta < 0:
            raise ValueError("beta must be >= 0")

        if self.kl_adj_factor < 0:
            raise ValueError("kl_adj_factor must be >= 0")
        if self.kl_adj_factor_max < self.kl_adj_factor:
            raise ValueError("kl_adj_factor_max must be >= kl_adj_factor")
        if self.kl_adj_update_factor <= 0:
            raise ValueError("kl_adj_update_factor must be > 0")
        if self.running_window < 2:
            raise ValueError("running_window must be >= 2")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.image_size <= 0:
            raise ValueError("image_size must be > 0")
        if not (0 < self.val_split < 1):
            raise ValueError("val_split must be between 0 and 1")
        if self.shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be > 0")

        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if self.base_filters <= 0:
            raise ValueError("base_filters must be > 0")
        if any(x <= 0 for x in self.filter_factors):
            raise ValueError("all filter_factors must be > 0")
        if self.encode_dense_units <= 0:
            raise ValueError("encode_dense_units must be > 0")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be > 0")

        if self.snapshot_every <= 0:
            raise ValueError("snapshot_every must be > 0")
        if self.train_preview_count < 0:
            raise ValueError("train_preview_count must be >= 0")
        if self.valid_preview_count < 0:
            raise ValueError("valid_preview_count must be >= 0")