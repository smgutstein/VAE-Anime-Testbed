from pathlib import Path
import logging
import pickle


class TrainingMonitor:
    """
    Owns training-time metric buffers and persistence of stats artifacts.
    Preserves existing on-disk formats expected by analysis code.
    """

    def __init__(self, stats_dir, snapshot_every):
        self.stats_dir = Path(stats_dir)
        self.snapshot_every = int(snapshot_every)

        self.recon_loss_list = []
        self.kl_loss_list = []
        self.adj_kl_factor_list = []

        self.mu_list = []
        self.log_var_list = []
        self.mu_var_list = []
        self.log_var_var_list = []

        self.loss_file = None
        self.f_loss_lists = None
        self.f_mu = None
        self.f_var = None


    def open(self):
        self.loss_file = open(self.stats_dir / "losses_file.txt", "w")
        self.f_loss_lists = open(self.stats_dir / "loss_lists.pkl", "wb")
        self.f_mu = open(self.stats_dir / "mu_log_var_lists.pkl", "wb")
        self.f_var = open(self.stats_dir / "mu_log_var_lists2.pkl", "wb")

        self.loss_file.write(
            "epoch -- step -- recon_loss -- kl_loss -- kl_adj_factor  "
            "bounce_test1 bounce_test2 num_maxes max_factor min_factor window_len\n"
        )
        return self

    def close(self):
        try:
            self.flush_partial_buffers()
        finally:
            for fh in (self.loss_file, self.f_loss_lists, self.f_mu, self.f_var):
                if fh is not None:
                    fh.close()

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

        self.loss_file.write(
            f"{epoch} -- {step} -- {loss_recon:.4f} -- {loss_kl:.4e} -- "
            f"{curr_kl_weight:.4e}  {test1} {test2} {num_maxes} "
            f"{max_kl_weight_seen} {min_kl_weight_seen} {window_len}\n"
        )

    def record_losses(self, curr_loss_recon, curr_loss_kl, curr_kl_weight):
        self.recon_loss_list.append(curr_loss_recon)
        self.kl_loss_list.append(curr_loss_kl)
        self.adj_kl_factor_list.append(curr_kl_weight)

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
            or self.adj_kl_factor_list
            or self.mu_list
            or self.log_var_list
            or self.mu_var_list
            or self.log_var_var_list
        ):
            return

        pickle.dump(
            [self.recon_loss_list, self.kl_loss_list, self.adj_kl_factor_list],
            self.f_loss_lists,
        )
        pickle.dump([self.mu_list, self.log_var_list], self.f_mu)
        pickle.dump([self.mu_var_list, self.log_var_var_list], self.f_var)

        self.loss_file.flush()
        self.f_loss_lists.flush()
        self.f_mu.flush()
        self.f_var.flush()

        self.recon_loss_list.clear()
        self.kl_loss_list.clear()
        self.adj_kl_factor_list.clear()
        self.mu_list.clear()
        self.log_var_list.clear()
        self.mu_var_list.clear()
        self.log_var_var_list.clear()

        logging.info("Flushed partial stats buffers")