import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import EngFormatter
from pathlib import Path
from PIL import Image
from time import time, ctime
from tqdm import tqdm
from utils import is_config_file
from utils import read_config_file

class AnalyzeResults():
    def __init__(self, config_file="config.ini"):

        # Load the config file
        assert Path(config_file).exists(), "Config file does not exist"
        assert is_config_file(config_file), "Invalid config file"
        
        self.config_file = config_file

        # Set the output directories
        self.get_output_dir()
        self.raw_image_dir = self.output_dir / "raw_images"

        self.stats_dir = self.output_dir / "stats" 
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        self.raw_log_var_graphs_dir = self.stats_dir / "raw_log_var_graphs"
        self.raw_log_var_graphs_dir.mkdir(parents=True, exist_ok=True)

        self.raw_mu_graphs_dir = self.stats_dir / "raw_mu_graphs"
        self.raw_mu_graphs_dir.mkdir(parents=True, exist_ok=True)

        self.movies_dir = self.output_dir / "movies"
        self.movies_dir.mkdir(parents=True, exist_ok=True)

    def get_output_dir(self): 
        config = read_config_file(self.config_file)
        self.parent_dir = config.get('Output_Parameters', 'parent_dir')
        expt_dirs = [x.name for x in Path('./expts').iterdir() 
                     if x.is_dir() and x.name.startswith('expt')]
        expt_dirs.sort(key=lambda x:int(x[5:]))
        curr_expt_dir = expt_dirs[-1]
        self.output_dir = Path('./expts') / curr_expt_dir
        
    def get_analysis_dir(self):
        self.load_config_file()
        return self.parent_dir / "analysis"

    def make_images_movie(self):

        # Function used to converted epoch-step labeling
        # of frames to frame numbers
        def frame_num(file_path):
            file_name = file_path.name
            epoch = int(file_name.split("_")[3])
            step = int(file_name.split("_")[4][4:-4])
            frame_num = 100*epoch + step
            return frame_num
        
        frames = [file for file in self.raw_image_dir.iterdir() 
                  if str(file.name)[:14] == "image_at_epoch"]
        frames.sort(key=frame_num)

        # Make movie
        movie_name = 'vae_movie.mp4'
        writer = imageio.get_writer(self.movies_dir / movie_name, fps=10)
        for file in tqdm(frames, desc='Making movie for ' + movie_name[:-4]):
            im = imageio.imread(file)
            writer.append_data(im)
        writer.close()
        
    def compare_recon_kl_losses(self):

        with open(self.stats_dir / Path('loss_lists.pkl'), 'rb') as f:
            recon_loss_list, kl_loss_list, _ = pickle.load(f)

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

        with open(self.stats_dir / Path('loss_lists.pkl'), 'rb') as f:
            recon_loss_list, kl_loss_list, adj_kl_factor_list = pickle.load(f)
        num_pts = len(recon_loss_list)

        # Create an EngFormatter object with desired precision
        formatter = EngFormatter(places=0, unit='')  # Adjust places and unit as needed


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
        #axes[0][1].yaxis.set_major_formatter(formatter)
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
        axes[1][1].yaxis.tick_right()
        axes[1][1].yaxis.set_label_position("right")
        axes[1][1].plot(range(num_pts), kl_loss_list,
                        label="kl loss", color='#ff7f0e')

        plt.savefig(self.stats_dir / Path('Recon_KL_Comp_2.png'))

    

    def make_paretoish_graph(self):

        # Read file with recon and kl losses
        with open(self.stats_dir / Path('losses_file.txt'),'r') as f:
            fl = f.readlines()

        recon_pts=[]
        kl_pts = []
        for curr_line in fl[1:]:
            data = [x.strip() for x in curr_line.split('--')]
            recon_pts.append(float(data[2]))
            kl_pts.append(float(data[3]))

        fig, ax = plt.subplots()

        skip_pts = int(.10*len(recon_pts))
        
        # create a color map
        colors = np.arange(len(recon_pts[skip_pts:]))

        # create a scatter plot on the axes with colors indicating the order
        sc = ax.scatter(recon_pts[skip_pts:], kl_pts[skip_pts:],
                        s=1, c=colors, cmap='winter')

        # Give the plot a title and labels
        ax.set_xlabel('Recon Loss')
        ax.set_ylabel('KL Loss')
        ax.set_title('Pareto-ish Graph')

        # add a colorbar
        color_bar = fig.colorbar(sc)
        color_bar.set_label("Pt Number")

        # Save graph
        plt.savefig(self.stats_dir / Path('Paretoish.png'))     

    def make_singleton_graphs(self):
        self.make_paretoish_graph()
        self.compare_recon_kl_losses()  
        self.compare_recon_kl_losses2()

    def make_mu_log_var_movie(self, log_var_graph=True):

        with open(self.stats_dir / Path('mu_log_var_lists.pkl'),'rb') as f:
            mu, log_var = pickle.load(f)

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
            ax.set_ylabel('Log Var')
            ax.axis([0,512, lower_lim, upper_lim])
            ax.set_title('Sample'+ str(ctr))
            ax.scatter(range(512), plot_data, s=3, color='cadetblue')
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

    def make_paretoish_movie(self):

        # Read file with recon and kl losses
        with open(self.stats_dir / Path('losses_file.txt'),'r') as f:
            fl = f.readlines()
            
        recon_pts = []
        kl_pts = []
        for curr_line in fl[1:]:  # Skipping the header
            data = [x.strip() for x in curr_line.split('--')]
            recon_pts.append(float(data[2]))
            kl_pts.append(float(data[3]))
            
        # Create a writer object
        writer = imageio.get_writer(self.movies_dir / "paretoish.mp4", fps=5)
        
        num_points = len(fl[1:])
        min_x, max_x = np.percentile(np.array(recon_pts),[0,95])
        min_y, max_y = np.percentile(np.array(kl_pts),[0,95])
        frame_len = int(0.05 * num_points)
        frame_delta = int(0.1 * frame_len)
        num_frames = int((num_points - frame_len)/frame_delta) + 1

        for idx in tqdm(range(num_frames+1),
                        desc='Making movie for paretoish'):

            # Find start & stop data points for this frame
            start = idx * frame_delta
            stop = min(idx * frame_delta + frame_len, num_points)

            # Slice out the data points for this frame
            r_pts = recon_pts[start:stop]
            k_pts = kl_pts[start:stop]
            
            # Create a figure
            fig, ax = plt.subplots()

            # create a color map
            colors = np.arange(len(r_pts))

            # create a scatter plot on the axes with colors indicating the order
            sc = ax.scatter(r_pts, k_pts, s=1,
                            c=colors, cmap='winter')
            ax.set_xlabel('Recon Loss')
            ax.set_ylabel('KL Loss')
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            # add a colorbar
            color_bar = fig.colorbar(sc)
            color_bar.set_label("Pt Number")

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set some params for training & output dir.')
    parser.add_argument('-c', '--config_file', type=str,
                        default='config.ini', help='Config file')
    parser.add_argument('-s', '--stat_graphs', action='store_true',
                        help='Get mu & sigma graphs and movie')
    args = parser.parse_args()
    
    ar = AnalyzeResults(args.config_file)
    ar.compare_recon_kl_losses()    
    ar.compare_recon_kl_losses2()
    ar.make_paretoish_graph()
    ar.make_images_movie()
    ar.make_log_var_movie()
    if args.stat_graphs:
        ar.make_mu_log_var_graphs()
    
