import logging
import math
import pickle

from pathlib import Path

import numpy as np


LOSS_TEXT_FILE = "losses_file.txt"
LOSS_LISTS_FILE = "loss_lists.pkl"
MU_LOG_VAR_FILE = "mu_log_var_lists.pkl"
MU_LOG_VAR_VAR_FILE = "mu_log_var_lists2.pkl"


def require_artifact(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _iter_pickled_objects(pickle_file):
    """
    Yield each object stored sequentially in a pickle file.
    """
    pickle_file = Path(pickle_file)
    with open(pickle_file, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def read_loss_file_points_io(loss_file):
    """
    Read recon/KL datapoints from stats/losses_file.txt.

    Current expected text format is written by
    TrainingMonitor.record_text_line(), i.e.

        epoch -- step -- recon_loss -- kl_loss -- kl_adj_factor ...

    Returns:
        fl              : filtered raw lines (header + valid data lines)
        recon_pts       : list[float]
        kl_pts          : list[float]
        kl_adj_factor_pts : list[float]
    """
    loss_file = require_artifact(loss_file, "loss text file")

    with open(loss_file, "r") as f:
        fl = f.readlines()

    # Remove lines with nan or inf to match current analysis behavior
    fl = [x for x in fl if ("nan" not in x.lower()) and ("inf" not in x.lower())]

    recon_pts = []
    kl_pts = []
    kl_adj_factor_pts = []

    for curr_line in fl[1:]:  # skip header
        data = [x.strip() for x in curr_line.split("--")]
        if len(data) >= 5:
            recon_pts.append(float(data[2]))
            kl_pts.append(float(data[3]))
            kl_adj_factor_pts.append(float(data[4].split()[0]))
        else:
            logging.warning(f"Last line of losses_file incomplete: {data}")
            break

    return fl, recon_pts, kl_pts, kl_adj_factor_pts


def read_loss_lists(stats_dir):
    """
    Read aggregated recon/KL/KL-factor data from stats/loss_lists.pkl.

    Current expected pickle chunk format is:
        [recon_loss_list, kl_loss_list, adj_kl_factor_list]

    Returns:
        recon_loss_list, kl_loss_list, adj_kl_factor_list
    """
    stats_dir = Path(stats_dir)
    loss_lists_file = require_artifact(
        stats_dir / LOSS_LISTS_FILE,
        "loss list pickle",
    )

    recon_loss_list = []
    kl_loss_list = []
    adj_kl_factor_list = []

    try:
        for data in _iter_pickled_objects(loss_lists_file):
            if isinstance(data, list) and len(data) >= 3:
                recon_loss_list += data[0]
                kl_loss_list += data[1]
                adj_kl_factor_list += data[2]
            else:
                logging.warning(f"Unexpected object in {loss_lists_file}: {type(data)}")
    except Exception as e:
        logging.error(f"Error reading {loss_lists_file}: {e}")
        raise

    recon_loss_list = [
        x for x in recon_loss_list
        if (not math.isnan(x)) and (not math.isinf(x))
    ]
    kl_loss_list = [
        x for x in kl_loss_list
        if (not math.isnan(x)) and (not math.isinf(x))
    ]
    adj_kl_factor_list = [
        x for x in adj_kl_factor_list
        if (not math.isnan(x)) and (not math.isinf(x))
    ]

    temp = min(len(recon_loss_list), len(kl_loss_list), len(adj_kl_factor_list))
    recon_loss_list = recon_loss_list[:temp]
    kl_loss_list = kl_loss_list[:temp]
    adj_kl_factor_list = adj_kl_factor_list[:temp]

    return recon_loss_list, kl_loss_list, adj_kl_factor_list


def read_mu_log_var_lists(stats_dir):
    """
    Read latent mean stats from stats/mu_log_var_lists.pkl.

    Current expected pickle chunk format is:
        [mu_list, log_var_list]

    Returns:
        mu, log_var
    """
    stats_dir = Path(stats_dir)
    mu_log_var_file = require_artifact(
        stats_dir / MU_LOG_VAR_FILE,
        "mu/log_var pickle",
    )

    mu = []
    log_var = []

    try:
        for data in _iter_pickled_objects(mu_log_var_file):
            if isinstance(data, list) and len(data) >= 2:
                mu += data[0]
                log_var += data[1]
            else:
                logging.warning(f"Unexpected object in {mu_log_var_file}: {type(data)}")
    except Exception as e:
        logging.error(f"Error reading {mu_log_var_file}: {e}")
        raise

    log_var = [
        x for x in log_var
        if (not np.isnan(x.numpy()).any()) and (not np.isinf(x.numpy()).any())
    ]
    mu = [
        x for x in mu
        if (not np.isnan(x.numpy()).any()) and (not np.isinf(x.numpy()).any())
    ]

    temp = min(len(log_var), len(mu))
    log_var = log_var[:temp]
    mu = mu[:temp]

    return mu, log_var


def read_mu_log_var_var_lists(stats_dir):
    """
    Read latent variance-of-stat stats from stats/mu_log_var_lists2.pkl.

    Current expected pickle chunk format is:
        [mu_var_list, log_var_var_list]

    Returns:
        mu_var, log_var_var
    """
    stats_dir = Path(stats_dir)
    mu_log_var_var_file = require_artifact(
        stats_dir / MU_LOG_VAR_VAR_FILE,
        "mu/log_var variance pickle",
    )

    mu_var = []
    log_var_var = []

    try:
        for data in _iter_pickled_objects(mu_log_var_var_file):
            if isinstance(data, list) and len(data) >= 2:
                mu_var += data[0]
                log_var_var += data[1]
            else:
                logging.warning(
                    f"Unexpected object in {mu_log_var_var_file}: {type(data)}"
                )
    except Exception as e:
        logging.error(f"Error reading {mu_log_var_var_file}: {e}")
        raise

    mu_var = [
        x for x in mu_var
        if (not np.isnan(x.numpy()).any()) and (not np.isinf(x.numpy()).any())
    ]
    log_var_var = [
        x for x in log_var_var
        if (not np.isnan(x.numpy()).any()) and (not np.isinf(x.numpy()).any())
    ]

    temp = min(len(mu_var), len(log_var_var))
    mu_var = mu_var[:temp]
    log_var_var = log_var_var[:temp]

    return mu_var, log_var_var