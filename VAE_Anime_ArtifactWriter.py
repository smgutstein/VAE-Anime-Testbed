import pickle
from pathlib import Path
from VAE_Anime_Artifacts import (
    LOSS_TEXT_FILE, LOSS_EVENTS_FILE, LATENT_STATS_FILE, LATENT_VAR_STATS_FILE,
    LossEventChunk, LatentMeanChunk, LatentVarChunk
)


class ArtifactWriter:
    def __init__(self, stats_dir):
        self.stats_dir = Path(stats_dir)
        self.loss_text_fh = None
        self.loss_events_fh = None
        self.latent_stats_fh = None
        self.latent_var_stats_fh = None

    def open(self):
        self.loss_text_fh = open(self.stats_dir / LOSS_TEXT_FILE, "w")
        self.loss_events_fh = open(self.stats_dir / LOSS_EVENTS_FILE, "wb")
        self.latent_stats_fh = open(self.stats_dir / LATENT_STATS_FILE, "wb")
        self.latent_var_stats_fh = open(self.stats_dir / LATENT_VAR_STATS_FILE, "wb")
        return self

    def write_loss_text_header(self):
        self.loss_text_fh.write(
            "epoch -- step -- recon_loss -- kl_loss -- kl_weight  "
            "bounce_test1 bounce_test2 num_maxes max_factor min_factor window_len "
            "kl_jump_ratio max_log_var min_log_var max_grad_norm\n"
        )

    def write_loss_text_line(
        self,
        epoch,
        step,
        loss_recon,
        loss_kl,
        curr_kl_weight,
        test1,
        test2,
        num_maxes,
        max_kl_weight_seen,
        min_kl_weight_seen,
        window_len,
        kl_jump_ratio,
        max_log_var,
        min_log_var,
        max_grad_norm,
    ):
        self.loss_text_fh.write(
            f"{epoch} -- {step} -- {float(loss_recon):.4f} -- {float(loss_kl):.4e} -- "
            f"{float(curr_kl_weight):.4e}  {test1} {test2} {num_maxes} "
            f"{max_kl_weight_seen} {min_kl_weight_seen} {window_len} "
            f"{float(kl_jump_ratio):.4e} {float(max_log_var):.4e} "
            f"{float(min_log_var):.4e} {float(max_grad_norm):.4e}\n"
        )



    def write_loss_chunk(self, recon_loss, kl_loss, kl_weight):
        chunk = LossEventChunk(
            recon_loss=list(recon_loss),
            kl_loss=list(kl_loss),
            kl_weight=list(kl_weight),
        )
        pickle.dump(chunk, self.loss_events_fh)

    def write_latent_chunk(self, mu, log_var):
        chunk = LatentMeanChunk(
            mu=list(mu),
            log_var=list(log_var),
        )
        pickle.dump(chunk, self.latent_stats_fh)

    def write_latent_var_chunk(self, mu_var, log_var_var):
        chunk = LatentVarChunk(
            mu_var=list(mu_var),
            log_var_var=list(log_var_var),
        )
        pickle.dump(chunk, self.latent_var_stats_fh)

    def flush(self):
        for fh in (
            self.loss_text_fh,
            self.loss_events_fh,
            self.latent_stats_fh,
            self.latent_var_stats_fh,
        ):
            if fh is not None:
                fh.flush()

    def close(self):
        for fh in (
            self.loss_text_fh,
            self.loss_events_fh,
            self.latent_stats_fh,
            self.latent_var_stats_fh,
        ):
            if fh is not None:
                fh.close()