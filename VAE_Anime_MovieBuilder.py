import imageio.v2 as imageio
import logging
import matplotlib.pyplot as plt
import numpy as np
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from VAE_Anime_ArtifactReader import ArtifactReader


class VAEMovieBuilder:
    """
    Builds movies from saved training/analysis frames.
    """

    def __init__(self, raw_image_dir, stats_dir, movies_dir):
        self.raw_image_dir = Path(raw_image_dir)
        self.stats_dir = Path(stats_dir)
        self.movies_dir = Path(movies_dir)
        self.reader = ArtifactReader(self.stats_dir)

    def _require_artifact(self, path, description):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
        return path

    def make_images_movie(self):
        """
        Build the training snapshot movie from raw snapshot PNGs.
        Expected filenames:
            image_at_epoch_0001_step0000.png
        """

        def frame_num(file_path):
            file_name = file_path.name
            match = re.match(r"^image_at_epoch_(\d+)_step(\d+)\.png$", file_name)
            if match is None:
                raise ValueError(f"Unexpected frame filename format: {file_name}")
            epoch = int(match.group(1))
            step = int(match.group(2))
            return 100 * epoch + step

        self._require_artifact(self.raw_image_dir, "raw image directory")

        frames = [
            file for file in self.raw_image_dir.iterdir()
            if re.match(r"^image_at_epoch_\d+_step\d+\.png$", file.name)
        ]

        if len(frames) == 0:
            raise RuntimeError(f"No movie frames found in {self.raw_image_dir}")

        frames.sort(key=frame_num)

        movie_name = "vae_movie.mp4"
        outpath = self.movies_dir / movie_name
        writer = imageio.get_writer(outpath, fps=10)

        for file in tqdm(frames, desc=f"Making movie for {movie_name[:-4]}"):
            im = imageio.imread(file)
            writer.append_data(im)

        writer.close()
        logging.info(f"Saved {outpath}")


    def make_mu_log_var_movie(self, log_var_graph=True):
        """
        Build latent-stat movie from structured artifacts.

        If log_var_graph is True, writes movies/log_var.mp4.
        Otherwise writes movies/mu.mp4.
        """
        series = self.reader.read_latent_series()
        mu, log_var = series.mu, series.log_var

        if log_var_graph:
            data = log_var
            graph_name = "log_var.mp4"
        else:
            data = mu
            graph_name = "mu.mp4"

        if len(data) == 0:
            raise RuntimeError(f"No latent-stat data found in {self.stats_dir}")

        # Normalize to numpy arrays for percentile calculations and plotting.
        data_np = [
            x.numpy() if hasattr(x, "numpy") else np.asarray(x)
            for x in data
        ]

        lower_lim = np.percentile(np.percentile(data_np, 2, axis=1), 2)
        upper_lim = np.percentile(np.percentile(data_np, 98, axis=1), 99)

        outpath = self.movies_dir / graph_name
        writer = imageio.get_writer(outpath, fps=20)

        for ctr, curr_data in enumerate(
            tqdm(data_np, desc="Making movie for " + graph_name[:-4])
        ):
            plot_data = np.sort(curr_data)

            fig, ax = plt.subplots()
            ax.set_xlabel("Ordered Indices")
            ax.set_ylabel(graph_name[:-4])
            ax.axis([0, len(plot_data), lower_lim, upper_lim])
            ax.set_title("Sample" + str(ctr))
            ax.scatter(range(len(plot_data)), plot_data, s=3, color="cadetblue")
            ax.axhline(y=0, color="lightsteelblue")

            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            image = Image.frombytes(
                "RGBA",
                canvas.get_width_height(),
                bytes(buf),
                "raw",
                "RGBA",
                0,
                1,
            )

            writer.append_data(np.array(image))
            plt.close(fig)

        writer.close()
        logging.info(f"Saved {outpath}")
