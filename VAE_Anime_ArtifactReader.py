from dataclasses import dataclass
import logging
import math
import pickle
from pathlib import Path

import numpy as np

from VAE_Anime_Artifacts import (
    LOSS_EVENTS_FILE,
    LATENT_STATS_FILE,
    LATENT_VAR_STATS_FILE,
    LATENT_KL_STATS_FILE,
    LossEventChunk,
    LatentMeanChunk,
    LatentVarChunk,
    LatentKLChunk,
)


# Legacy filenames kept here so ArtifactReader can read both new and old runs.
LEGACY_LOSS_LISTS_FILE = "loss_lists.pkl"
LEGACY_MU_LOG_VAR_FILE = "mu_log_var_lists.pkl"
LEGACY_MU_LOG_VAR_VAR_FILE = "mu_log_var_lists2.pkl"


@dataclass
class LossSeries:
    recon_loss: list[float]
    kl_loss: list[float]
    kl_weight: list[float]
    recon_ssim: list[float]
    active_latent_dims: list[int]


@dataclass
class LatentSeries:
    mu: list
    log_var: list


@dataclass
class LatentVarSeries:
    mu_var: list
    log_var_var: list


@dataclass
class LatentKLSeries:
    kl_per_dim: list


class ArtifactReader:
    def __init__(self, stats_dir):
        self.stats_dir = Path(stats_dir)

    def _iter_pickled_objects(self, pickle_file):
        pickle_file = Path(pickle_file)
        with open(pickle_file, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def _as_numpy(self, x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    def _is_valid_numeric(self, x):
        arr = self._as_numpy(x)
        return (not np.isnan(arr).any()) and (not np.isinf(arr).any())

    def _clean_scalar_list(self, values):
        out = []
        for x in values:
            try:
                x = float(x)
            except Exception:
                logging.warning("Could not convert scalar value to float: %r", x)
                continue
            if not math.isnan(x) and not math.isinf(x):
                out.append(x)
        return out

    def _clean_tensorlike_list(self, values):
        return [x for x in values if self._is_valid_numeric(x)]

    def _clean_int_list(self, values):
        out = []
        for x in values:
            try:
                out.append(int(x))
            except Exception:
                logging.warning("Could not convert integer value: %r", x)
        return out

    def read_loss_series(self):
        new_path = self.stats_dir / LOSS_EVENTS_FILE
        if new_path.exists():
            return self._read_new_loss_series(new_path)

        legacy_path = self.stats_dir / LEGACY_LOSS_LISTS_FILE
        if legacy_path.exists():
            return self._read_legacy_loss_series(legacy_path)

        raise FileNotFoundError(
            f"Missing loss artifacts: neither {new_path} nor {legacy_path} exists"
        )

    def read_latent_series(self):
        new_path = self.stats_dir / LATENT_STATS_FILE
        if new_path.exists():
            return self._read_new_latent_series(new_path)

        legacy_path = self.stats_dir / LEGACY_MU_LOG_VAR_FILE
        if legacy_path.exists():
            return self._read_legacy_latent_series(legacy_path)

        raise FileNotFoundError(
            f"Missing latent artifacts: neither {new_path} nor {legacy_path} exists"
        )

    def read_latent_var_series(self):
        new_path = self.stats_dir / LATENT_VAR_STATS_FILE
        if new_path.exists():
            return self._read_new_latent_var_series(new_path)

        legacy_path = self.stats_dir / LEGACY_MU_LOG_VAR_VAR_FILE
        if legacy_path.exists():
            return self._read_legacy_latent_var_series(legacy_path)

        raise FileNotFoundError(
            f"Missing latent variance artifacts: neither {new_path} nor {legacy_path} exists"
        )


    def read_latent_kl_series(self):
        new_path = self.stats_dir / LATENT_KL_STATS_FILE
        if new_path.exists():
            return self._read_new_latent_kl_series(new_path)

        raise FileNotFoundError(f"Missing latent KL artifact: {new_path}")

    def _read_new_loss_series(self, path):
        recon_loss = []
        kl_loss = []
        kl_weight = []
        recon_ssim = []
        active_latent_dims = []

        for obj in self._iter_pickled_objects(path):
            if not isinstance(obj, LossEventChunk):
                logging.warning("Unexpected object in %s: %s", path, type(obj))
                continue
            recon_loss += obj.recon_loss
            kl_loss += obj.kl_loss
            kl_weight += obj.kl_weight
            recon_ssim += getattr(obj, "recon_ssim", [])
            active_latent_dims += getattr(obj, "active_latent_dims", [])

        recon_loss = self._clean_scalar_list(recon_loss)
        kl_loss = self._clean_scalar_list(kl_loss)
        kl_weight = self._clean_scalar_list(kl_weight)
        recon_ssim = self._clean_scalar_list(recon_ssim)
        active_latent_dims = self._clean_int_list(active_latent_dims)

        n = min(len(recon_loss), len(kl_loss), len(kl_weight))
        metric_n = min(n, len(recon_ssim), len(active_latent_dims))
        return LossSeries(
            recon_loss=recon_loss[:n],
            kl_loss=kl_loss[:n],
            kl_weight=kl_weight[:n],
            recon_ssim=recon_ssim[:metric_n],
            active_latent_dims=active_latent_dims[:metric_n],
        )

    def _read_new_latent_series(self, path):
        mu = []
        log_var = []

        for obj in self._iter_pickled_objects(path):
            if not isinstance(obj, LatentMeanChunk):
                logging.warning("Unexpected object in %s: %s", path, type(obj))
                continue
            mu += obj.mu
            log_var += obj.log_var

        mu = self._clean_tensorlike_list(mu)
        log_var = self._clean_tensorlike_list(log_var)

        n = min(len(mu), len(log_var))
        return LatentSeries(mu=mu[:n], log_var=log_var[:n])

    def _read_new_latent_var_series(self, path):
        mu_var = []
        log_var_var = []

        for obj in self._iter_pickled_objects(path):
            if not isinstance(obj, LatentVarChunk):
                logging.warning("Unexpected object in %s: %s", path, type(obj))
                continue
            mu_var += obj.mu_var
            log_var_var += obj.log_var_var

        mu_var = self._clean_tensorlike_list(mu_var)
        log_var_var = self._clean_tensorlike_list(log_var_var)

        n = min(len(mu_var), len(log_var_var))
        return LatentVarSeries(mu_var=mu_var[:n], log_var_var=log_var_var[:n])


    def _read_new_latent_kl_series(self, path):
        kl_per_dim = []

        for obj in self._iter_pickled_objects(path):
            if not isinstance(obj, LatentKLChunk):
                logging.warning("Unexpected object in %s: %s", path, type(obj))
                continue
            kl_per_dim += obj.kl_per_dim

        return LatentKLSeries(
            kl_per_dim=self._clean_tensorlike_list(kl_per_dim),
        )

    def _read_legacy_loss_series(self, path):
        recon_loss = []
        kl_loss = []
        kl_weight = []

        for data in self._iter_pickled_objects(path):
            if isinstance(data, list) and len(data) >= 3:
                recon_loss += data[0]
                kl_loss += data[1]
                kl_weight += data[2]
            else:
                logging.warning("Unexpected legacy object in %s: %s", path, type(data))

        recon_loss = self._clean_scalar_list(recon_loss)
        kl_loss = self._clean_scalar_list(kl_loss)
        kl_weight = self._clean_scalar_list(kl_weight)

        n = min(len(recon_loss), len(kl_loss), len(kl_weight))
        return LossSeries(
            recon_loss=recon_loss[:n],
            kl_loss=kl_loss[:n],
            kl_weight=kl_weight[:n],
            recon_ssim=[],
            active_latent_dims=[],
        )

    def _read_legacy_latent_series(self, path):
        mu = []
        log_var = []

        for data in self._iter_pickled_objects(path):
            if isinstance(data, list) and len(data) >= 2:
                mu += data[0]
                log_var += data[1]
            else:
                logging.warning("Unexpected legacy object in %s: %s", path, type(data))

        mu = self._clean_tensorlike_list(mu)
        log_var = self._clean_tensorlike_list(log_var)

        n = min(len(mu), len(log_var))
        return LatentSeries(mu=mu[:n], log_var=log_var[:n])

    def _read_legacy_latent_var_series(self, path):
        mu_var = []
        log_var_var = []

        for data in self._iter_pickled_objects(path):
            if isinstance(data, list) and len(data) >= 2:
                mu_var += data[0]
                log_var_var += data[1]
            else:
                logging.warning("Unexpected legacy object in %s: %s", path, type(data))

        mu_var = self._clean_tensorlike_list(mu_var)
        log_var_var = self._clean_tensorlike_list(log_var_var)

        n = min(len(mu_var), len(log_var_var))
        return LatentVarSeries(mu_var=mu_var[:n], log_var_var=log_var_var[:n])