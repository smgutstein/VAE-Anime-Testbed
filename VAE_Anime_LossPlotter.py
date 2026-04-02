from pathlib import Path

import matplotlib.pyplot as plt

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