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

        self.raw_log_var_graphs_dir = self.stats_dir / "raw_log_var_graphs"
        self.raw_log_var_graphs_dir.mkdir(parents=True, exist_ok=True)

        self.raw_mu_graphs_dir = self.stats_dir / "raw_mu_graphs"
        self.raw_mu_graphs_dir.mkdir(parents=True, exist_ok=True)

        self.movies_dir = self.output_dir / "movies"
        self.movies_dir.mkdir(parents=True, exist_ok=True)

        

    def get_parent_dir(self):
        config = read_config_file(self.config_file)
        return Path(config.get('Output_Parameters', 'parent_dir'))

    def require_artifact(self, path, description):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
        return path


    def make_images_movie(self):

        # Function used to converted epoch-step labeling
        # of frames to frame numbers
        def frame_num(file_path):
            file_name = file_path.name
            match = re.match(r"^image_at_epoch_(\d+)_step(\d+)\.png$", file_name)
            if match is None:
                raise ValueError(f"Unexpected frame filename format: {file_name}")
            epoch = int(match.group(1))
            step = int(match.group(2))

            frame_num = 100*epoch + step
            return frame_num
        
        self.require_artifact(self.raw_image_dir, "raw image directory")
        frames = [file for file in self.raw_image_dir.iterdir() 
                  if re.match(r"^image_at_epoch_\d+_step\d+\.png$", file.name)]
        if len(frames) == 0:
            raise RuntimeError(f"No movie frames found in {self.raw_image_dir}")
        else:
            frames.sort(key=frame_num)

        # Make movie
        movie_name = 'vae_movie.mp4'
        writer = imageio.get_writer(self.movies_dir / movie_name, fps=10)
        for file in tqdm(frames, desc='Making movie for ' + movie_name[:-4]):
            im = imageio.imread(file)
            writer.append_data(im)
        writer.close()
        logging.info(f"Saved {self.movies_dir / movie_name}")

    def get_recon_kl_results(self):
        return read_loss_lists(self.stats_dir)
 
    
    def read_loss_file_points(self):
        """Read recon/KL points from stats/losses_file.txt.

        Returns:
            recon_pts, kl_pts
        """
        fl, recon_pts, kl_pts, _ = read_loss_file_points_io(
            self.stats_dir / Path("losses_file.txt")
        )
        return fl, recon_pts, kl_pts

        
    def compare_recon_kl_losses(self):

        recon_loss_list, kl_loss_list, _ = self.get_recon_kl_results()
        if len(recon_loss_list) == 0 or len(kl_loss_list) == 0:
            raise RuntimeError(f"No reconstruction/KL points found in {self.stats_dir}")


        fig, axes = plt.subplots(3)  # Create a figure containing a single axes.
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_yscale('log')
        axes[0].plot(range(len(recon_loss_list)), recon_loss_list,
                     label="recon\n loss")
        axes[0].plot(range(len(recon_loss_list)), kl_loss_list,
                     label="kl\n loss")
        axes[0].set_ylim([.1,1000])
        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                       prop={'size': 6})

        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Recon Loss')
        axes[1].plot(range(len(recon_loss_list)), recon_loss_list,
                     label="recon\n loss")
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                       prop={'size': 6})

        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('KL Loss')
        axes[2].plot(range(len(recon_loss_list)), kl_loss_list,
                     label="kl\n loss", color='#ff7f0e')
        axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                       prop={'size': 6})

        plt.savefig(self.stats_dir / Path('Recon_KL_Comp_1.png'))


    def compare_recon_kl_losses2(self):

        recon_loss_list, kl_loss_list, adj_kl_factor_list = self.get_recon_kl_results()
        if len(recon_loss_list) == 0 or len(kl_loss_list) == 0:
            raise RuntimeError(f"No reconstruction/KL points found in {self.stats_dir}")

        num_pts = len(recon_loss_list)


        fig, axes = plt.subplots(2,2)  
        # Log scale plot of both losses
        axes[0][0].set_xlabel('Iteration',fontsize=8, labelpad=-2)
        axes[0][0].set_ylabel('Loss')
        axes[0][0].set_yscale('log')
        axes[0][0].plot(range(num_pts), recon_loss_list, label="recon loss")
        axes[0][0].plot(range(num_pts), kl_loss_list, label="kl loss")
        axes[0][0].set_ylim([.0001,1000])

        # Plot adjusted kl loss
        axes[0][1].set_xlabel('Iteration',fontsize=8, labelpad=-2)
        axes[0][1].set_ylabel('Adj KL Loss')
        axes[0][1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[0][1].plot(range(num_pts), adj_kl_factor_list,
                        label="adj kl factor", color='#ff7f0e')
        axes[0][1].yaxis.tick_right()
        axes[0][1].yaxis.set_label_position("right")

        # Plot recon loss
        axes[1][0].set_xlabel('Iteration')
        axes[1][0].set_ylabel('Recon Loss')
        axes[1][0].plot(range(num_pts), recon_loss_list, label="recon loss")

        # Plot kl loss  
        axes[1][1].set_xlabel('Iteration')
        axes[1][1].set_ylabel('KL Loss')
        axes[1][1].set_yscale('log')
        axes[1][1].yaxis.tick_right()
        axes[1][1].yaxis.set_label_position("right")
        axes[1][1].plot(range(num_pts), kl_loss_list,
                        label="kl loss", color='#ff7f0e')

        plt.savefig(self.stats_dir / Path('Recon_KL_Comp_2.png'))

    

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
        self.make_final_mu_log_var_graphs()
        self.compare_recon_kl_losses()  
        self.compare_recon_kl_losses2()

    def get_recon_kl_results(self):
        series = self.reader.read_loss_series()
        return series.recon_loss, series.kl_loss, series.kl_weight

    def get_mu_log_var_results(self):
        series = self.reader.read_latent_series()
        return series.mu, series.log_var
    
    def make_final_mu_log_var_graphs(self):
        mu, log_var = self.get_mu_log_var_results()
        graph_list = [(mu[-1], 'mu.png'), (log_var[-1], 'log_var.png')]
        
        for data, graph_name in graph_list:
            plot_data = np.sort(data.numpy())
            fig, ax = plt.subplots()
            ax.set_xlabel('Ordered Indices')
            ax.set_ylabel(graph_name[:-4])
            ax.set_title('Final '+ graph_name[:-4])
            ax.scatter(range(len(data)), plot_data, s=3, color='cadetblue')
            ax.axhline(y=0, color='lightsteelblue')
            plt.savefig(self.stats_dir / Path(graph_name))
            logging.info(f" Saved {self.stats_dir / Path(graph_name)}")

    def make_mu_log_var_movie(self, log_var_graph=True):

        mu, log_var = self.get_mu_log_var_results()
        
        if log_var_graph:
            data = log_var
            graph_name="log_var.mp4"
        else:
            data = mu
            graph_name="mu.mp4"

        lower_lim = np.percentile(np.percentile(data, 2, axis=1), 2)
        upper_lim = np.percentile(np.percentile(data, 98, axis=1), 99)

        writer = imageio.get_writer(self.movies_dir / graph_name, 
                                    fps=20)

        for ctr, curr_data in enumerate(tqdm(data, 
                                             desc='Making movie for ' + graph_name[:-4])):
            plot_data = np.sort(curr_data.numpy())
            fig, ax = plt.subplots()
            ax.set_xlabel('Ordered Indices')
            ax.set_ylabel(graph_name[:-4])
            ax.axis([0,len(plot_data), lower_lim, upper_lim])
            ax.set_title('Sample'+ str(ctr))
            ax.scatter(range(len(plot_data)), plot_data, s=3, color='cadetblue')
            ax.axhline(y=0, color='lightsteelblue')

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
        
        # Close Writer
        writer.close()
        logging.info(f"Saved {self.movies_dir / graph_name}")

    def test_pareto_curve(self):
        # Read file with recon and kl losses
        _, recon_pts, kl_pts = self.read_loss_file_points()
 

        self.pareto_front.clear_front()
        self.pareto_front.add_points(recon_pts, kl_pts)
        #self.pareto_curve = self.pareto_front.get_smooth_pareto_curve()
        return recon_pts, kl_pts


    def make_paretoish_movie(self):

        # Read file with recon and kl losses
        fl, recon_pts, kl_pts = self.read_loss_file_points()
 
        skip_pts = int(.05*len(recon_pts))   
        recon_pts = recon_pts[skip_pts:]
        kl_pts = kl_pts[skip_pts:]
        # Create a writer object
        writer = imageio.get_writer(self.movies_dir / "paretoish.mp4", fps=5)
        
        num_points = len(fl[1:])
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
    ar.make_images_movie()
    ar.make_paretoish_movie()
    ar.make_pareto_curve_graph()
    if args.stat_graphs:
        ar.make_mu_log_var_movie(log_var_graph=True)
        ar.make_mu_log_var_movie(log_var_graph=False)

    
