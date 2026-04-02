from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

LOSS_TEXT_FILE = "losses_file.txt"
LOSS_EVENTS_FILE = "loss_events.pkl"
LATENT_STATS_FILE = "latent_stats.pkl"
LATENT_VAR_STATS_FILE = "latent_var_stats.pkl"

@dataclass
class LossEventChunk:
    schema_version: int = SCHEMA_VERSION
    recon_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)
    kl_weight: list[float] = field(default_factory=list)

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