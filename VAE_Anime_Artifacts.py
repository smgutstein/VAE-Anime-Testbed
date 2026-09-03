from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2

LOSS_TEXT_FILE = "losses_file.txt"
VAL_LOSS_TEXT_FILE = "val_losses_file.txt"
LOSS_EVENTS_FILE = "loss_events.pkl"
LATENT_STATS_FILE = "latent_stats.pkl"
LATENT_VAR_STATS_FILE = "latent_var_stats.pkl"
LATENT_KL_STATS_FILE = "latent_kl_stats.pkl"
VAL_LATENT_STATS_FILE = "val_latent_stats.pkl"

@dataclass
class LossEventChunk:
    schema_version: int = SCHEMA_VERSION
    recon_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)
    kl_weight: list[float] = field(default_factory=list)
    recon_ssim: list[float] = field(default_factory=list)
    active_latent_dims: list[int] = field(default_factory=list)

@dataclass
class LatentMeanChunk:
    schema_version: int = SCHEMA_VERSION
    mu: list = field(default_factory=list)
    log_var: list = field(default_factory=list)

@dataclass
class LatentVarChunk:
    schema_version: int = SCHEMA_VERSION
    mu_var: list = field(default_factory=list)
    log_var_var: list = field(default_factory=list)


@dataclass
class LatentKLChunk:
    schema_version: int = SCHEMA_VERSION
    kl_per_dim: list = field(default_factory=list)


@dataclass
class ValLatentChunk:
    """Per-dimension latent statistics from one validation evaluation.

    agg_post_var_per_dim is Var(mu_d) + E[sigma^2_d] over the whole
    validation split, which equals 1.0 for every dimension when the
    aggregate posterior matches a standard normal prior. The scalar
    summaries in val_losses_file.txt can sit near 1.0 while individual
    dimensions are far from it, so the per-dimension array is kept.
    """

    schema_version: int = SCHEMA_VERSION
    epoch: int = 0
    kl_per_dim: list = field(default_factory=list)
    agg_post_var_per_dim: list = field(default_factory=list)
