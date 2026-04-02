from pathlib import Path

import matplotlib.pyplot as plt

from VAE_Anime_ArtifactReader import ArtifactReader


class VAELatentStatsPlotter:
    """
    Reads structured latent-stat artifacts and creates latent summary plots.
    """

    def __init__(self, stats_dir):
        self.stats_dir = Path(stats_dir)
        self.reader = ArtifactReader(self.stats_dir)

    def make_singleton_graphs(self):
        self.make_paretoish_graph()
        self.latent_plotter.make_final_mu_log_var_graphs()
        self.loss_plotter.compare_recon_kl_losses()
        self.loss_plotter.compare_recon_kl_losses2()