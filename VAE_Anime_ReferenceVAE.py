"""Save the best-validation-SSIM VAE as a candidate reference model."""

from dataclasses import asdict
import json
import logging
from pathlib import Path
import shutil



class BestSSIMReferenceSaver:
    """Keep the VAE checkpoint with the highest deterministic validation SSIM."""

    def __init__(self, output_dir, cfg, resume=False):
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        # Reference VAEs live outside the experiment tree:
        #   <project>/expts/expt_N/
        #   <project>/ref_vae/expt_N/
        self.ref_root = self.output_dir.parent.parent / "ref_vae"
        self.ref_dir = self.ref_root / self.output_dir.name
        self.best_ssim = float("-inf")
        # Only a resumed run may inherit a prior best SSIM.  A fresh run can
        # reuse an experiment number (e.g. after the latest expt dir was
        # deleted), and must not be blocked by that deleted run's metadata.
        metadata_path = self.ref_dir / "metadata.json"
        if resume and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.best_ssim = float(
                    metadata["selected_state"]["validation_ssim_mu"]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                logging.warning(
                    "Could not recover prior reference-VAE best SSIM from %s",
                    metadata_path,
                )

    def consider(self, *, vae, val_result, epoch, step, kl_weight):
        """Save ``vae`` if ``val_result.ssim_mu`` is the best seen so far."""
        current_ssim = float(val_result.ssim_mu)
        if current_ssim <= self.best_ssim:
            return False

        self.ref_dir.mkdir(parents=True, exist_ok=True)

        weights_path = self.ref_dir / "reference_vae.weights.h5"

        # Save only weights.  The copied config and provenance files are
        # sufficient to rebuild the VAE architecture from project source.
        vae.vae_net.save_weights(weights_path, overwrite=True)

        config_src = Path(self.cfg.config_file)
        shutil.copy2(config_src, self.ref_dir / "config.ini")

        notes_src = self.output_dir / "Notes.txt"
        if notes_src.exists():
            shutil.copy2(notes_src, self.ref_dir / "Notes.txt")

        metadata = {
            "schema_version": 1,
            "purpose": "candidate reference VAE for image-feature distribution comparisons",
            "selection_criterion": "maximum validation ssim_mu (decoder(mu) reconstruction)",
            "selected_state": {
                "epoch": int(epoch),
                "step": int(step),
                "validation_ssim_mu": current_ssim,
                "kl_weight": float(kl_weight),
            },
            "validation_metrics": asdict(val_result),
            "training": {
                "loss_policy": self.cfg.loss_policy,
                "beta": self.cfg.beta,
                "initial_kl_weight": self.cfg.initial_kl_weight,
                "max_kl_weight": self.cfg.max_kl_weight,
                "kl_weight_update_factor": self.cfg.kl_weight_update_factor,
                "running_window": self.cfg.running_window,
                "learning_rate": self.cfg.learning_rate,
                "seed": self.cfg.seed,
                "deterministic": self.cfg.deterministic,
            },
            "model": {
                "image_size": self.cfg.image_size,
                "latent_dim": self.cfg.latent_dim,
                "base_filters": self.cfg.base_filters,
                "filter_factors": list(self.cfg.filter_factors),
                "encode_dense_units": self.cfg.encode_dense_units,
                "kernel_size": self.cfg.kernel_size,
            },
            "data": {
                "data_dir": str(self.cfg.data_dir),
                "batch_size": self.cfg.batch_size,
                "val_split": self.cfg.val_split,
                "shuffle_buffer": self.cfg.shuffle_buffer,
                "train_drop_remainder": self.cfg.train_drop_remainder,
            },
            "files": {
                "vae_weights": weights_path.name,
                "config": "config.ini",
                "notes": "Notes.txt" if notes_src.exists() else None,
            },
        }

        def _json_default(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            if hasattr(value, "item"):
                return value.item()
            raise TypeError(
                f"Object of type {value.__class__.__name__} "
                "is not JSON serializable"
            )

        with open(self.ref_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)

        self.best_ssim = current_ssim
        logging.info(
            "Saved new reference-VAE candidate at epoch %d step %d "
            "(validation ssim_mu=%.6f) to %s",
            epoch,
            step,
            current_ssim,
            self.ref_dir,
        )
        return True
