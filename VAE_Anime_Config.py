from dataclasses import dataclass
from pathlib import Path

from utils import (
    get_data_dir,
    get_parent_dir,
    get_seed_and_determinism,
    load_and_validate_config,
    parse_bool,
)


def resolve_config_path(
    config_file: str | Path,
    default_config_dir: str | Path = "configs",
) -> Path:
    """
    Resolve a config path with minimal user friction.

    Resolution order:
    1. Use the provided path as-is if it exists.
    2. Try the current working directory's config dir.
    3. Try the repo/module directory's config dir.
    4. Raise FileNotFoundError with a helpful message.
    """
    raw_path = Path(config_file)

    if raw_path.is_file():
        return raw_path

    here = Path(__file__).resolve().parent
    fallback_candidates = [
        Path(default_config_dir) / raw_path,
        here / default_config_dir / raw_path,
    ]

    for candidate in fallback_candidates:
        if candidate.is_file():
            return candidate

    tried = [str(raw_path)] + [str(p) for p in fallback_candidates]
    raise FileNotFoundError(
        f"Could not find config file '{config_file}'. Tried: {tried}"
    )


@dataclass(frozen=True)
class TrainerConfig:
    config_file: Path

    # Training
    epochs: int
    learning_rate: float
    loss_policy: str
    beta: float | None
    initial_kl_weight: float | None
    max_kl_weight: float | None
    kl_weight_update_factor: float | None
    running_window: int | None

    # Safety / trip-wire
    max_grad_norm: float | None
    step_guard_kl_jump_ratio_threshold: float
    step_guard_kl_abs_threshold: float
    step_guard_max_log_var_threshold: float
    tripwire_lr_backoff: float
    tripwire_lr_floor: float
    max_consecutive_tripwires: int

    # Output
    parent_dir: Path
    save_net: bool
    save_ref_vae: bool
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
    checkpoint_every_epochs: int
    flush_every_epochs: int
    validate_every_epochs: int
    train_preview_count: int
    valid_preview_count: int
    take_initial_snapshot: bool
    run_analysis: bool
    make_mu_log_var_movies: bool
    active_dim_kl_threshold: float

    # Reproducibility
    seed: int
    deterministic: bool

    @classmethod
    def from_file(cls, config_file="config.ini"):
        config_path = resolve_config_path(Path(config_file))
        config = load_and_validate_config(config_path)
        safety_section = "Safety_Parameters"

        def _get_kl_float(new_name, old_name):
            section = "Training_Parameters"
            if config.has_option(section, new_name):
                return config.getfloat(section, new_name)
            if config.has_option(section, old_name):
                return config.getfloat(section, old_name)
            raise ValueError(
                f"Missing [{section}] {new_name} (or legacy {old_name})"
            )

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
            beta=(
                config.getfloat("Training_Parameters", "beta")
                if config.get("Training_Parameters", "loss_policy") == "fixed_beta"
                else None
            ),            
            initial_kl_weight=(
                _get_kl_float("initial_kl_weight", "kl_adj_factor")
                if config.get("Training_Parameters", "loss_policy") == "adaptive_kl"
                else None
            ),
            max_kl_weight=(
                _get_kl_float("max_kl_weight", "kl_adj_factor_max")
                if config.get("Training_Parameters", "loss_policy") == "adaptive_kl"
                else None
            ),
            kl_weight_update_factor=(
                _get_kl_float("kl_weight_update_factor", "kl_adj_update_factor")
                if config.get("Training_Parameters", "loss_policy") == "adaptive_kl"
                else None
            ),
            running_window=(
                config.getint("Training_Parameters", "running_window")
                if config.get("Training_Parameters", "loss_policy") == "adaptive_kl"
                else None
            ),

            # Safety / trip-wire
            max_grad_norm=(
                config.getfloat(safety_section, "max_grad_norm")
                if config.has_option(safety_section, "max_grad_norm")
                else None
            ),
            step_guard_kl_jump_ratio_threshold=config.getfloat(
                safety_section, "step_guard_kl_jump_ratio_threshold", fallback=100.0
            ),
            step_guard_kl_abs_threshold=config.getfloat(
                safety_section, "step_guard_kl_abs_threshold", fallback=1e6
            ),
            step_guard_max_log_var_threshold=config.getfloat(
                safety_section, "step_guard_max_log_var_threshold", fallback=20.0
            ),
            tripwire_lr_backoff=config.getfloat(
                safety_section, "tripwire_lr_backoff", fallback=0.5
            ),
            tripwire_lr_floor=config.getfloat(
                safety_section, "tripwire_lr_floor", fallback=1e-6
            ),
            max_consecutive_tripwires=config.getint(
                safety_section, "max_consecutive_tripwires", fallback=3
            ),

            # Output
            parent_dir=get_parent_dir(config),
            save_net=parse_bool(config.get("Output_Parameters", "save_net"), "save_net"),
            save_ref_vae=parse_bool(
                config.get("Output_Parameters", "save_ref_vae", fallback="false"),
                "save_ref_vae",
            ),
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
            checkpoint_every_epochs=config.getint(
                "Monitoring_Parameters",
                "checkpoint_every_epochs",
                fallback=-1,
            ),
            flush_every_epochs=config.getint(
                "Monitoring_Parameters", "flush_every_epochs", fallback=100
            ),
            validate_every_epochs=config.getint(
                "Monitoring_Parameters", "validate_every_epochs", fallback=25
            ),
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
            active_dim_kl_threshold=config.getfloat(
                "Monitoring_Parameters",
                "active_dim_kl_threshold",
                fallback=1e-2,
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

        if self.loss_policy == "fixed_beta":
            if self.beta is None:
                raise ValueError("fixed_beta requires beta")
            if self.beta < 0:
                raise ValueError("beta must be >= 0")

        if self.loss_policy == "adaptive_kl":
            if self.initial_kl_weight is None:
                raise ValueError("adaptive_kl requires initial_kl_weight")
            if self.max_kl_weight is None:
                raise ValueError("adaptive_kl requires max_kl_weight")
            if self.kl_weight_update_factor is None:
                raise ValueError("adaptive_kl requires kl_weight_update_factor")
            if self.running_window is None:
                raise ValueError("adaptive_kl requires running_window")

            if self.initial_kl_weight < 0:
                raise ValueError("initial_kl_weight must be >= 0")
            if self.max_kl_weight < self.initial_kl_weight:
                raise ValueError("max_kl_weight must be >= initial_kl_weight")
            if self.kl_weight_update_factor <= 0:
                raise ValueError("kl_weight_update_factor must be > 0")
            if self.running_window < 2:
                raise ValueError("running_window must be >= 2")
            
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when provided")
        if self.step_guard_kl_jump_ratio_threshold <= 0:
            raise ValueError("step_guard_kl_jump_ratio_threshold must be > 0")
        if self.step_guard_kl_abs_threshold <= 0:
            raise ValueError("step_guard_kl_abs_threshold must be > 0")
        if self.step_guard_max_log_var_threshold <= 0:
            raise ValueError("step_guard_max_log_var_threshold must be > 0")

            
        if self.tripwire_lr_backoff <= 0:
            raise ValueError("tripwire_lr_backoff must be > 0")
        if self.tripwire_lr_floor <= 0:
            raise ValueError("tripwire_lr_floor must be > 0")
        if self.max_consecutive_tripwires <= 0:
            raise ValueError("max_consecutive_tripwires must be > 0")
        if self.active_dim_kl_threshold <= 0:
            raise ValueError("active_dim_kl_threshold must be > 0")

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
        if self.checkpoint_every_epochs == 0 or self.checkpoint_every_epochs < -1:
            raise ValueError(
                "checkpoint_every_epochs must be -1 (disabled) or > 0"
            )
        if self.flush_every_epochs <= 0:
            raise ValueError("flush_every_epochs must be > 0")
        if self.validate_every_epochs <= 0:
            raise ValueError("validate_every_epochs must be > 0")
        if self.train_preview_count < 0:
            raise ValueError("train_preview_count must be >= 0")
        if self.valid_preview_count < 0:
            raise ValueError("valid_preview_count must be >= 0")