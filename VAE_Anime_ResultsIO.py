import logging
from pathlib import Path

from VAE_Anime_ArtifactReader import ArtifactReader


LOSS_TEXT_FILE = "losses_file.txt"


def require_artifact(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def read_loss_file_points_io(loss_file):
    """
    Read recon/KL datapoints from stats/losses_file.txt.

    Current expected text format is written by
    TrainingMonitor.record_text_line(), i.e.

        epoch -- step -- recon_loss -- kl_loss -- kl_weight ...

    Returns:
        fl                : filtered raw lines (header + valid data lines)
        recon_pts         : list[float]
        kl_pts            : list[float]
        kl_weight_pts : list[float]
    """
    loss_file = require_artifact(loss_file, "loss text file")

    with open(loss_file, "r") as f:
        fl = f.readlines()

    # Remove lines with nan or inf to match current analysis behavior
    fl = [x for x in fl if ("nan" not in x.lower()) and ("inf" not in x.lower())]

    recon_pts = []
    kl_pts = []
    kl_weight_pts = []

    for curr_line in fl[1:]:  # skip header
        data = [x.strip() for x in curr_line.split("--")]
        if len(data) >= 5:
            recon_pts.append(float(data[2]))
            kl_pts.append(float(data[3]))
            kl_weight_pts.append(float(data[4].split()[0]))
        else:
            logging.warning("Last line of losses_file incomplete: %s", data)
            break

    return fl, recon_pts, kl_pts, kl_weight_pts


def read_loss_lists(stats_dir):
    """
    Compatibility wrapper over ArtifactReader.

    Returns:
        recon_loss_list, kl_loss_list, kl_weight_list
    """
    series = ArtifactReader(stats_dir).read_loss_series()
    return series.recon_loss, series.kl_loss, series.kl_weight


def read_mu_log_var_lists(stats_dir):
    """
    Compatibility wrapper over ArtifactReader.

    Returns:
        mu, log_var
    """
    series = ArtifactReader(stats_dir).read_latent_series()
    return series.mu, series.log_var


def read_mu_log_var_var_lists(stats_dir):
    """
    Compatibility wrapper over ArtifactReader.

    Returns:
        mu_var, log_var_var
    """
    series = ArtifactReader(stats_dir).read_latent_var_series()
    return series.mu_var, series.log_var_var