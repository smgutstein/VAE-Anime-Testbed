from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from VAE_Anime_ArtifactReader import ArtifactReader


class VAELossPlotter:
    """
    Reads structured loss artifacts and creates loss comparison plots.
    """

    def __init__(self, stats_dir):
        self.stats_dir = Path(stats_dir)
        self.reader = ArtifactReader(self.stats_dir)

    def get_recon_kl_results(self):
        series = self.reader.read_loss_series()
        return series.recon_loss, series.kl_loss, series.kl_weight

    def get_loss_diagnostic_results(self):
        """
        Return diagnostic loss series stored alongside the loss artifact.
        """
        series = self.reader.read_loss_series()
        return series.recon_loss, series.recon_ssim, series.active_latent_dims

    def get_latent_kl_results(self):
        """
        Return the per-dimension KL series.

        Each item in ``kl_per_dim`` corresponds to one recorded training
        iteration and contains one KL value per latent dimension.
        """
        series = self.reader.read_latent_kl_series()
        return series.kl_per_dim

    def compare_recon_kl_losses(self):
        recon_loss_list, kl_loss_list, _ = self.get_recon_kl_results()
        if len(recon_loss_list) == 0 or len(kl_loss_list) == 0:
            raise RuntimeError(f"No reconstruction/KL points found in {self.stats_dir}")

        fig, axes = plt.subplots(3)

        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_yscale("log")
        axes[0].plot(range(len(recon_loss_list)), recon_loss_list, label="recon\n loss")
        axes[0].plot(range(len(recon_loss_list)), kl_loss_list, label="kl\n loss")
        axes[0].set_ylim([0.1, 1000])
        axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6})

        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Recon Loss")
        axes[1].plot(range(len(recon_loss_list)), recon_loss_list, label="recon\n loss")
        axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6})

        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("KL Loss")
        axes[2].plot(range(len(recon_loss_list)), kl_loss_list, label="kl\n loss", color="#ff7f0e")
        axes[2].legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6})

        plt.savefig(self.stats_dir / Path("Recon_KL_Comp_1.png"))
        plt.close(fig)

    def compare_recon_kl_losses2(self):
        recon_loss_list, kl_loss_list, kl_weight_list = self.get_recon_kl_results()
        if len(recon_loss_list) == 0 or len(kl_loss_list) == 0:
            raise RuntimeError(f"No reconstruction/KL points found in {self.stats_dir}")

        num_pts = len(recon_loss_list)

        fig, axes = plt.subplots(2, 2)

        axes[0][0].set_xlabel("Iteration", fontsize=8, labelpad=-2)
        axes[0][0].set_ylabel("Loss")
        axes[0][0].set_yscale("log")
        axes[0][0].plot(range(num_pts), recon_loss_list, label="recon loss")
        axes[0][0].plot(range(num_pts), kl_loss_list, label="kl loss")
        axes[0][0].set_ylim([0.0001, 1000])

        axes[0][1].set_xlabel("Iteration", fontsize=8, labelpad=-2)
        axes[0][1].set_ylabel("KL Weight")
        axes[0][1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        axes[0][1].plot(range(num_pts), kl_weight_list, label="kl weight", color="#ff7f0e")
        axes[0][1].yaxis.tick_right()
        axes[0][1].yaxis.set_label_position("right")

        axes[1][0].set_xlabel("Iteration")
        axes[1][0].set_ylabel("Recon Loss")
        axes[1][0].plot(range(num_pts), recon_loss_list, label="recon loss")

        axes[1][1].set_xlabel("Iteration")
        axes[1][1].set_ylabel("KL Loss")
        axes[1][1].set_yscale("log")
        axes[1][1].yaxis.tick_right()
        axes[1][1].yaxis.set_label_position("right")
        axes[1][1].plot(range(num_pts), kl_loss_list, label="kl loss", color="#ff7f0e")

        plt.savefig(self.stats_dir / Path("Recon_KL_Comp_2.png"))
        plt.close(fig)

    def plot_active_latent_dims(self):
        """
        Plot the number of active latent dimensions over training iterations.
        """
        _, _, active_latent_dims = self.get_loss_diagnostic_results()
        if len(active_latent_dims) == 0:
            return False

        fig, ax = plt.subplots()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Active Latent Dimensions")
        ax.set_title("Active Latent Dimensions vs Iteration")
        ax.plot(range(len(active_latent_dims)), active_latent_dims, label="active latent dims")
        ax.legend(loc="best", prop={"size": 6})

        plt.savefig(self.stats_dir / Path("Active_Latent_Dims.png"))
        plt.close(fig)
        return True

    def plot_kl_per_dim(self):
        """
        Plot per-dimension KL values over training iterations.
        """
        try:
            kl_per_dim = self.get_latent_kl_results()
        except FileNotFoundError:
            return False

        if len(kl_per_dim) == 0:
            return False

        kl_arr = np.asarray(kl_per_dim, dtype=float)
        if kl_arr.ndim != 2 or kl_arr.shape[0] == 0 or kl_arr.shape[1] == 0:
            return False

        fig, ax = plt.subplots()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("KL per Dimension")
        ax.set_title("Per-Dimension KL vs Iteration")
        for dim_idx in range(kl_arr.shape[1]):
            ax.plot(range(kl_arr.shape[0]), kl_arr[:, dim_idx], label=f"z{dim_idx}")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 6})

        plt.savefig(self.stats_dir / Path("KL_Per_Dim.png"), bbox_inches="tight")
        plt.close(fig)
        return True

    def plot_recon_loss_vs_ssim(self):
        """
        Plot the relationship between reconstruction loss and SSIM.
        """
        recon_loss, recon_ssim, _ = self.get_loss_diagnostic_results()
        n = min(len(recon_loss), len(recon_ssim))
        if n == 0:
            return False

        fig, ax = plt.subplots()
        ax.set_xlabel("Recon Loss")
        ax.set_ylabel("Recon SSIM")
        ax.set_title("Recon Loss vs Recon SSIM")
        ax.scatter(recon_loss[:n], recon_ssim[:n], s=4)

        plt.savefig(self.stats_dir / Path("Recon_Loss_vs_SSIM.png"))
        plt.close(fig)
        return True

    def make_diagnostic_graphs(self):
        """
        Create all available diagnostic plots.

        Missing optional diagnostic artifacts are skipped so analysis remains
        usable for older runs that predate these metrics.
        """
        return {
            "active_latent_dims": self.plot_active_latent_dims(),
            "kl_per_dim": self.plot_kl_per_dim(),
            "recon_loss_vs_ssim": self.plot_recon_loss_vs_ssim(),
        }
