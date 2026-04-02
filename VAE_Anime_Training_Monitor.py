from pathlib import Path
import logging

from VAE_Anime_ArtifactWriter import ArtifactWriter


class TrainingMonitor:
    """
    Owns training-time metric buffers only.

    Persistence is delegated to ArtifactWriter so this class no longer
    knows artifact filenames or on-disk schemas.
    """

    def __init__(self, stats_dir, snapshot_every):
        self.stats_dir = Path(stats_dir)
        self.snapshot_every = int(snapshot_every)

        self.recon_loss_list = []
        self.kl_loss_list = []
        self.kl_weight_list = []

        self.mu_list = []
        self.log_var_list = []
        self.mu_var_list = []
        self.log_var_var_list = []

        self.writer = ArtifactWriter(self.stats_dir)

    def open(self):
        self.writer.open()
        self.writer.write_loss_text_header()
        return self

    def close(self):
        try:
            self.flush_partial_buffers()
        finally:
            self.writer.close()

    def flush(self):
        self.flush_partial_buffers()
        self.writer.flush()

    def is_snapshot_step(self, step):
        return step % self.snapshot_every == 0

    def record_text_line(
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
    ):
        self.writer.write_loss_text_line(
            epoch=epoch,
            step=step,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
            curr_kl_weight=curr_kl_weight,
            test1=test1,
            test2=test2,
            num_maxes=num_maxes,
            max_kl_weight_seen=max_kl_weight_seen,
            min_kl_weight_seen=min_kl_weight_seen,
            window_len=window_len,
        )

    def record_losses(self, curr_loss_recon, curr_loss_kl, curr_kl_weight):
        self.recon_loss_list.append(float(curr_loss_recon))
        self.kl_loss_list.append(float(curr_loss_kl))
        self.kl_weight_list.append(float(curr_kl_weight))

    def record_latent_stats(self, mu_mean, log_var_mean, mu_var, log_var_var):
        self.mu_list.append(mu_mean)
        self.log_var_list.append(log_var_mean)
        self.mu_var_list.append(mu_var)
        self.log_var_var_list.append(log_var_var)

    def maybe_flush_step(self, step):
        if self.is_snapshot_step(step):
            self.flush_partial_buffers()

    def flush_partial_buffers(self):
        if not (
            self.recon_loss_list
            or self.kl_loss_list
            or self.kl_weight_list
            or self.mu_list
            or self.log_var_list
            or self.mu_var_list
            or self.log_var_var_list
        ):
            return

        self.writer.write_loss_chunk(
            recon_loss=self.recon_loss_list,
            kl_loss=self.kl_loss_list,
            kl_weight=self.kl_weight_list,
        )
        self.writer.write_latent_chunk(
            mu=self.mu_list,
            log_var=self.log_var_list,
        )
        self.writer.write_latent_var_chunk(
            mu_var=self.mu_var_list,
            log_var_var=self.log_var_var_list,
        )
        self.writer.flush()

        self.recon_loss_list.clear()
        self.kl_loss_list.clear()
        self.kl_weight_list.clear()
        self.mu_list.clear()
        self.log_var_list.clear()
        self.mu_var_list.clear()
        self.log_var_var_list.clear()

        logging.info("Flushed partial stats buffers")