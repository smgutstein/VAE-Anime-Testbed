import argparse
import imageio.v2 as imageio
import logging
import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from utils import get_experiment_dir
from utils import get_latest_experiment_dir
from utils import setup_logging

from VAE_Anime_ArtifactReader import ArtifactReader
from VAE_Anime_Config import TrainerConfig
from VAE_Anime_LatentStatsPlotter import VAELatentStatsPlotter
from VAE_Anime_MovieBuilder import VAEMovieBuilder
from VAE_Anime_LossPlotter import VAELossPlotter
from VAE_Anime_ResultsIO import read_loss_file_points_io

from VAE_ParetoFront import ParetoFront


class AnalyzeResults():
    def __init__(self, config_file="config.ini", expt=-1):

        # Load the config file
        self.cfg = TrainerConfig.from_file(config_file)
        self.config_file = self.cfg.config_file
        self.parent_dir = self.cfg.parent_dir

        if expt != -1:
            # Get the output directory for the specified experiment
            self.output_dir = get_experiment_dir(self.parent_dir, expt)
        else:
            self.output_dir = get_latest_experiment_dir(self.parent_dir)
        self.pareto_front = ParetoFront()



        logging.info(f"Making graphs of results in {self.output_dir}")
        self.raw_image_dir = self.output_dir / "raw_images"
        self.raw_image_dir.mkdir(parents=True, exist_ok=True)

        self.stats_dir = self.output_dir / "stats" 
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.reader = ArtifactReader(self.stats_dir)
        self.loss_plotter = VAELossPlotter(self.stats_dir)
        self.latent_plotter = VAELatentStatsPlotter(self.stats_dir)

        self.raw_log_var_graphs_dir = self.stats_dir / "raw_log_var_graphs"
        self.raw_log_var_graphs_dir.mkdir(parents=True, exist_ok=True)

        self.raw_mu_graphs_dir = self.stats_dir / "raw_mu_graphs"
        self.raw_mu_graphs_dir.mkdir(parents=True, exist_ok=True)

        self.movies_dir = self.output_dir / "movies"
        self.movies_dir.mkdir(parents=True, exist_ok=True)

        self.movie_builder = VAEMovieBuilder(raw_image_dir=self.raw_image_dir,
                                             stats_dir=self.stats_dir,
                                             movies_dir=self.movies_dir)

    

    def require_artifact(self, path, description):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
        return path
 
    
    def read_loss_file_points(self):
        """
        Backward-compatible reader for Pareto-ish analysis.

        Prefer structured loss artifacts. Fall back to losses_file.txt only if
        structured artifacts are unavailable.
        """
        try:
            recon_pts, kl_pts, kl_weight_pts = self.read_loss_points()

            # Synthesize a minimal fl-compatible object so old callers that use
            # len(fl[1:]) do not break immediately.
            fl = ["structured_artifact_header\n"] + [
                f"{idx} -- {idx} -- {r} -- {k} -- {w}\n"
                for idx, (r, k, w) in enumerate(zip(recon_pts, kl_pts, kl_weight_pts))
            ]
            return fl, recon_pts, kl_pts

        except Exception as e:
            logging.warning(
                "Falling back to losses_file.txt for Pareto-ish analysis because "
                "structured loss read failed: %s",
                e,
            )
            fl, recon_pts, kl_pts, _ = read_loss_file_points_io(
                self.stats_dir / Path("losses_file.txt")
            )
            return fl, recon_pts, kl_pts
    
    def read_loss_points(self):
        """
        Preferred loss-point reader for Pareto-ish analysis.

        Returns:
            recon_pts, kl_pts, kl_weight_pts
        """
        series = self.reader.read_loss_series()
        return series.recon_loss, series.kl_loss, series.kl_weight
    
    def make_pareto_graph(self):

        # Read file with recon and kl losses
        _, recon_pts, kl_pts = self.read_loss_file_points()

        fig, ax = plt.subplots()

        skip_pts = int(.05*len(recon_pts))
        
        # create a color map
        colors = np.arange(len(recon_pts[skip_pts:]))

        # create a scatter plot on the axes with colors indicating the order
        sc = ax.scatter(recon_pts[skip_pts:], kl_pts[skip_pts:],
                        s=1, c=colors, cmap='cool')

        # Give the plot a title and labels
        ax.set_xlabel('Recon Loss')
        ax.set_ylabel('KL Loss')
        ax.set_title('Pareto Graph')

        # add a colorbar
        color_bar = fig.colorbar(sc)
        color_bar.set_label("Pt Number")

        # Save graph
        plt.savefig(self.stats_dir / Path('Pareto_Comparisons.png'))   
        logging.info(f"Saved {self.stats_dir / Path('Pareto_Comparisons.png')}")  

    def make_singleton_graphs(self):
        self.make_pareto_graph()
        self.latent_plotter.make_final_mu_log_var_graphs()
        self.loss_plotter.compare_recon_kl_losses()  
        self.loss_plotter.compare_recon_kl_losses2()
        self.loss_plotter.make_diagnostic_graphs()

    def get_recon_kl_results(self):
        series = self.reader.read_loss_series()
        return series.recon_loss, series.kl_loss, series.kl_weight

    def test_pareto_curve(self):
        # Read file with recon and kl losses
        _, recon_pts, kl_pts = self.read_loss_file_points()
 

        self.pareto_front.clear_front()
        self.pareto_front.add_points(recon_pts, kl_pts)
        #self.pareto_curve = self.pareto_front.get_smooth_pareto_curve()
        return recon_pts, kl_pts



    def _axis_limits_from_percentiles(self, points, lower_pct=0, upper_pct=99.9):
        """
        Return stable plot limits for a movie axis.

        Percentile-based limits keep a small number of extreme points from
        making the moving scatter plot unreadable. If all values are identical,
        add a small padding so Matplotlib does not warn about singular limits.
        """
        points = np.asarray(points, dtype=float)
        min_val, max_val = np.percentile(points, [lower_pct, upper_pct])

        if min_val == max_val:
            pad = 0.5 if min_val == 0 else abs(min_val) * 0.05
            min_val -= pad
            max_val += pad

        return min_val, max_val

    def _make_growing_scatter_movie(
        self,
        x_pts,
        y_pts,
        movie_name,
        description,
        xlabel,
        ylabel,
        title,
        y_log=False,
    ):
        """
        Make a growing scatter-plot movie, matching the Pareto movie style.

        Each frame contains all points up to the current stop point. Point color
        encodes temporal order, so the movie shows how the relationship evolves
        during training without implying that active-dimension values are
        monotonic or linearly connected.
        """
        x_pts = np.asarray(x_pts, dtype=float)
        y_pts = np.asarray(y_pts, dtype=float)
        n = min(len(x_pts), len(y_pts))
        if n == 0:
            logging.warning("Skipping %s movie: no points available", description)
            return False

        x_pts = x_pts[:n]
        y_pts = y_pts[:n]

        if y_log:
            valid_mask = y_pts > 0
            if not np.any(valid_mask):
                logging.warning(
                    "Skipping %s movie: log-scale y-axis has no positive points",
                    description,
                )
                return False
            x_pts = x_pts[valid_mask]
            y_pts = y_pts[valid_mask]
            n = len(x_pts)

        skip_pts = int(0.05 * n)
        x_pts = x_pts[skip_pts:]
        y_pts = y_pts[skip_pts:]
        num_points = len(x_pts)
        if num_points == 0:
            logging.warning("Skipping %s movie: no points remain after skipping warmup", description)
            return False

        min_x, max_x = self._axis_limits_from_percentiles(x_pts)
        min_y, max_y = self._axis_limits_from_percentiles(y_pts)

        frame_len = max(1, int(0.05 * num_points))
        frame_delta = max(1, int(0.1 * frame_len))
        num_frames = int((num_points - frame_len) / frame_delta) + 1

        outpath = self.movies_dir / movie_name
        writer = imageio.get_writer(outpath, fps=5)

        for idx in tqdm(range(num_frames + 1), desc=f"Making movie for {description}"):
            stop = min(idx * frame_delta + frame_len, num_points)
            if stop <= 0:
                continue

            x_frame = x_pts[:stop]
            y_frame = y_pts[:stop]
            colors = np.arange(len(x_frame))

            fig, ax = plt.subplots()
            scatter = ax.scatter(x_frame, y_frame, s=1, c=colors, cmap="cool")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            if y_log:
                ax.set_yscale("log")

            color_bar = fig.colorbar(scatter)
            color_bar.set_label("Pt Number")

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
        return True

    def make_active_dims_relationship_movies(self):
        """
        Make movies for active latent dimensions vs KL/reconstruction loss.

        These mirror ``make_pareto_movie``: each frame shows the accumulated
        scatter points up to that point in training, with point color serving as
        a proxy for iteration/time.
        """
        series = self.reader.read_loss_series()
        active_dims = series.active_latent_dims

        made_kl = self._make_growing_scatter_movie(
            active_dims,
            series.kl_loss,
            movie_name="active_dims_vs_kl_loss.mp4",
            description="active dims vs kl loss",
            xlabel="Active Latent Dimensions",
            ylabel="KL Loss",
            title="Active Latent Dimensions vs KL Loss",
            y_log=True,
        )
        made_recon = self._make_growing_scatter_movie(
            active_dims,
            series.recon_loss,
            movie_name="active_dims_vs_recon_loss.mp4",
            description="active dims vs recon loss",
            xlabel="Active Latent Dimensions",
            ylabel="Recon Loss",
            title="Active Latent Dimensions vs Recon Loss",
            y_log=False,
        )

        return {
            "active_dims_vs_kl_loss": made_kl,
            "active_dims_vs_recon_loss": made_recon,
        }

    def make_pareto_movie(self):

        # Read file with recon and kl losses
        _, recon_pts, kl_pts = self.read_loss_file_points()

        skip_pts = int(.05 * len(recon_pts))
        recon_pts = recon_pts[skip_pts:]
        kl_pts = kl_pts[skip_pts:]
        writer = imageio.get_writer(self.movies_dir / "pareto.mp4", fps=5)

        num_points = len(recon_pts)
        
        min_x, max_x = np.percentile(np.array(recon_pts),[0,99.9])
        min_y, max_y = np.percentile(np.array(kl_pts),[0,99.9])
        frame_len = max(1,int(0.05 * num_points))
        frame_delta = max(1,int(0.1 * frame_len))
        num_frames = int((num_points - frame_len)/frame_delta) + 1

        self.pareto_front.clear_front()
        self.pareto_front.add_points(recon_pts, kl_pts)

        pareto_curve = self.pareto_front.get_front()#self.pareto_front.get_smooth_pareto_curve()
        if pareto_curve:
            pareto_x = [pt[0] for pt in pareto_curve]
            pareto_y = [pt[1] for pt in pareto_curve]
        else:
            pareto_x = []
            pareto_y = []


        for idx in tqdm(range(num_frames+1),
                        desc='Making movie for pareto'):

            # Find start & stop data points for this frame
            start = idx * frame_delta
            stop = min(idx * frame_delta + frame_len, num_points)

            # Slice out the data points for this frame
            r_pts = recon_pts[0:stop]
            k_pts = kl_pts[0:stop]
            
            # Create a figure
            fig, ax = plt.subplots()

            # create a color map
            colors = np.arange(len(r_pts))


            # create a scatter plot with colors indicating temporal order
            ax.scatter(r_pts, k_pts, s=1,
                        c=colors, cmap='cool')
            
            # Plot Pareto curve (line + points)
            ax.plot(pareto_x, pareto_y, 
                    color='dodgerblue', label='Smoothed Pareto', linewidth=2)
            ax.scatter(pareto_x, pareto_y, 
                    color='darkslategray', s=10)  # show curve points
            
            ax.set_xlabel('Recon Loss')
            ax.set_ylabel('KL Loss')
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            ax.set_yscale('log')

            # Convert the figure to an image
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            image = Image.frombytes('RGBA', canvas.get_width_height(),
                                    bytes(buf), 'raw', 'RGBA', 0, 1)

            # Write the image to the movie file
            writer.append_data(np.array(image))

            # Close the figure
            plt.close(fig)
            
        # Close the writer
        writer.close()   
        
    def make_pareto_curve_graph(self):
        recon_loss_list, kl_loss_list, _ = self.get_recon_kl_results()
        self.pareto_front.clear_front()
        self.pareto_front.add_points(recon_loss_list, kl_loss_list)
        pareto_curve = np.array(self.pareto_front.get_front())#np.array(self.pareto_front.get_smooth_pareto_curve())

        fig, ax = plt.subplots()
        ax.set_xlabel('Recon Loss')
        ax.set_ylabel('KL Loss')
        ax.set_title('Pareto Curve')
        ax.set_yscale('log')

        # Plot Pareto curve (line + points)
        ax.plot(pareto_curve[:, 0], pareto_curve[:, 1], 
                color='dodgerblue', label='Smoothed Pareto', linewidth=2)
        ax.scatter(pareto_curve[:, 0], pareto_curve[:, 1], 
                   color='darkslategray', s=10)  # show curve points
        
        outpath = self.stats_dir / "ParetoCurve.png"
        plt.savefig(outpath)
        logging.info(f" Saved {str(outpath)}")

        outpath2 = self.stats_dir / "ParetoPoints.txt"
        with outpath2.open("w", encoding="utf-8") as f:
            f.write("      recon_loss         kl_loss\n")
            for recon_loss, kl_loss in pareto_curve:
                f.write(f"{recon_loss:.17g} {kl_loss:.17g}\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set some params for training & output dir.')
    parser.add_argument('-c', '--config_file', type=str,
                        default='config.ini', help='Config file')
    parser.add_argument('-s', '--stat_graphs', action='store_true',
                        help='Get mu & sigma graphs and movie')
    parser.add_argument('-e', '--expt', type=int, default=-1,
                        help='Get specified expt number. Default is last expt.')
    parser.add_argument("--log", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log)
    
    ar = AnalyzeResults(args.config_file, args.expt)
    ar.make_singleton_graphs()  
    ar.movie_builder.make_images_movie()
    ar.make_pareto_movie()
    ar.make_active_dims_relationship_movies()
    ar.make_pareto_curve_graph()
    if args.stat_graphs:
        ar.movie_builder.make_mu_log_var_movie(log_var_graph=True)
        ar.movie_builder.make_mu_log_var_movie(log_var_graph=False)

    
