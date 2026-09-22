from pathlib import Path
import logging

from VAE_Anime_ArtifactWriter import ArtifactWriter


class TrainingMonitor:
    """
    Owns training-time metric buffers only.

    Persistence is delegated to ArtifactWriter so this class no longer
    knows artifact filenames or on-disk schemas.
    """

    def __init__(self, stats_dir, snapshot_every, append=False):
        self.stats_dir = Path(stats_dir)
        self.snapshot_every = int(snapshot_every)

        self.recon_loss_list = []
        self.kl_loss_list = []
        self.kl_weight_list = []
        self.recon_ssim_list = []
        self.active_latent_dims_list = []

        self.mu_list = []
        self.log_var_list = []
        self.mu_var_list = []
        self.log_var_var_list = []
        self.kl_per_dim_list = []

        self.append = bool(append)
        self.writer = ArtifactWriter(self.stats_dir, append=self.append)

    def open(self):
        self.writer.open()
        if not self.append:
            self.writer.write_loss_text_header()
            self.writer.write_val_loss_text_header()
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
        kl_jump_ratio,
        max_log_var,
        min_log_var,
        max_grad_norm,
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
            kl_jump_ratio=kl_jump_ratio,
            max_log_var=max_log_var,
            min_log_var=min_log_var,
            max_grad_norm=max_grad_norm,
        )

    def record_losses(
        self,
        curr_loss_recon,
        curr_loss_kl,
        curr_kl_weight,
        recon_ssim=None,
        active_latent_dims=None,
    ):
        self.recon_loss_list.append(float(curr_loss_recon))
        self.kl_loss_list.append(float(curr_loss_kl))
        self.kl_weight_list.append(float(curr_kl_weight))

        if recon_ssim is not None:
            self.recon_ssim_list.append(float(recon_ssim))
        if active_latent_dims is not None:
            self.active_latent_dims_list.append(int(active_latent_dims))

    def record_latent_stats(
        self,
        mu_mean,
        log_var_mean,
        mu_var,
        log_var_var,
        kl_per_dim=None,
    ):
        self.mu_list.append(mu_mean)
        self.log_var_list.append(log_var_mean)
        self.mu_var_list.append(mu_var)
        self.log_var_var_list.append(log_var_var)

        if kl_per_dim is not None:
            self.kl_per_dim_list.append(kl_per_dim)

    def record_validation(self, epoch, step, kl_weight, result):
        """Write one held-out evaluation row. `result` is a ValidationResult."""
        self.writer.write_val_loss_text_line(
            epoch=epoch,
            step=step,
            recon_mu=result.recon_mu,
            recon_sampled=result.recon_sampled,
            loss_kl=result.kl_loss,
            kl_sum=result.kl_sum,
            curr_kl_weight=kl_weight,
            ssim_mu=result.ssim_mu,
            ssim_sampled=result.ssim_sampled,
            active_latent_dims=result.active_latent_dims,
            agg_post_mean=result.agg_post_var_mean,
            agg_post_min=result.agg_post_var_min,
            agg_post_max=result.agg_post_var_max,
            n_images=result.n_images,
        )
        self.writer.write_val_latent_chunk(
            epoch=epoch,
            kl_per_dim=result.kl_per_dim,
            agg_post_var_per_dim=result.agg_post_var_per_dim,
        )

    def maybe_flush_step(self, step):
        if self.is_snapshot_step(step):
            self.flush_partial_buffers()

    def flush_partial_buffers(self):
        if not (
            self.recon_loss_list
            or self.kl_loss_list
            or self.kl_weight_list
            or self.recon_ssim_list
            or self.active_latent_dims_list
            or self.mu_list
            or self.log_var_list
            or self.mu_var_list
            or self.log_var_var_list
            or self.kl_per_dim_list
        ):
            return

        self.writer.write_loss_chunk(
            recon_loss=self.recon_loss_list,
            kl_loss=self.kl_loss_list,
            kl_weight=self.kl_weight_list,
            recon_ssim=self.recon_ssim_list,
            active_latent_dims=self.active_latent_dims_list,
        )
        self.writer.write_latent_chunk(
            mu=self.mu_list,
            log_var=self.log_var_list,
        )
        self.writer.write_latent_var_chunk(
            mu_var=self.mu_var_list,
            log_var_var=self.log_var_var_list,
        )
        self.writer.write_latent_kl_chunk(
            kl_per_dim=self.kl_per_dim_list,
        )
        self.writer.flush()

        self.recon_loss_list.clear()
        self.kl_loss_list.clear()
        self.kl_weight_list.clear()
        self.recon_ssim_list.clear()
        self.active_latent_dims_list.clear()
        self.mu_list.clear()
        self.log_var_list.clear()
        self.mu_var_list.clear()
        self.log_var_var_list.clear()
        self.kl_per_dim_list.clear()

        logging.info("Flushed partial stats buffers")
