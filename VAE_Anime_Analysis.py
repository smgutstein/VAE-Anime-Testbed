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
from utils import find_nan_or_inf_index
from utils import get_experiment_dir
from utils import get_latest_experiment_dir
from utils import is_config_file
from utils import read_config_file
from utils import setup_logging

from VAE_Anime_ArtifactReader import ArtifactReader
from VAE_Anime_LatentStatsPlotter import VAELatentStatsPlotter
from VAE_Anime_MovieBuilder import VAEMovieBuilder
from VAE_Anime_LossPlotter import VAELossPlotter
from VAE_Anime_ResultsIO import read_loss_file_points_io

from VAE_ParetoFront import ParetoFront


class AnalyzeResults():
    def __init__(self, config_file="config.ini", expt=-1):

        # Load the config file
        assert Path(config_file).exists(), f"{config_file} does not exist"
        assert is_config_file(config_file), f"{config_file} is invalid config file"
        self.config_file = config_file
        self.parent_dir = self.get_parent_dir()

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

        

    def get_parent_dir(self):
        config = read_config_file(self.config_file)
        return Path(config.get('Output_Parameters', 'parent_dir'))

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
    
    def make_paretoish_graph(self):

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
        ax.set_title('Pareto-ish Graph')

        # add a colorbar
        color_bar = fig.colorbar(sc)
        color_bar.set_label("Pt Number")

        # Save graph
        plt.savefig(self.stats_dir / Path('Paretoish.png'))   
        logging.info(f"Saved {self.stats_dir / Path('Paretoish.png')}")  

    def make_singleton_graphs(self):
        self.make_paretoish_graph()
        self.latent_plotter.make_final_mu_log_var_graphs()
        self.loss_plotter.compare_recon_kl_losses()  
        self.loss_plotter.compare_recon_kl_losses2()

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


    def make_paretoish_movie(self):

        # Read file with recon and kl losses
        _, recon_pts, kl_pts = self.read_loss_file_points()

        skip_pts = int(.05 * len(recon_pts))
        recon_pts = recon_pts[skip_pts:]
        kl_pts = kl_pts[skip_pts:]
        writer = imageio.get_writer(self.movies_dir / "paretoish.mp4", fps=5)

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
                        desc='Making movie for paretoish'):

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
    ar.make_paretoish_movie()
    ar.make_pareto_curve_graph()
    if args.stat_graphs:
        ar.movie_builder.make_mu_log_var_movie(log_var_graph=True)
        ar.movie_builder.make_mu_log_var_movie(log_var_graph=False)

    
