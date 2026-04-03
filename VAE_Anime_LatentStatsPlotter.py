from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from VAE_Anime_ArtifactReader import ArtifactReader


class VAELatentStatsPlotter:
    """
    Reads structured latent-stat artifacts and creates latent summary plots.
    """

    def __init__(self, stats_dir):
        self.stats_dir = Path(stats_dir)
        self.reader = ArtifactReader(self.stats_dir)

    def get_mu_log_var_results(self):
        series = self.reader.read_latent_series()
        return series.mu, series.log_var

    def make_final_mu_log_var_graphs(self):
        mu, log_var = self.get_mu_log_var_results()
        graph_list = [(mu[-1], "mu.png"), (log_var[-1], "log_var.png")]

        for data, graph_name in graph_list:
            plot_data = np.sort(data.numpy() if hasattr(data, "numpy") else np.asarray(data))
            fig, ax = plt.subplots()
            ax.set_xlabel("Ordered Indices")
            ax.set_ylabel(graph_name[:-4])
            ax.set_title("Final " + graph_name[:-4])
            ax.scatter(range(len(plot_data)), plot_data, s=3, color="cadetblue")
            ax.axhline(y=0, color="lightsteelblue")
            plt.savefig(self.stats_dir / Path(graph_name))
            plt.close(fig)