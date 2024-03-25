import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import pickle

from pathlib import Path
from tqdm import tqdm

class AnalyzeResults():
    def __init__(self, output_dir="scratch_output"):

        # Set the output directories
        self.output_dir = Path(output_dir)
        self.raw_image_dir = self.output_dir / "raw_images"

        self.stats_dir = self.output_dir / "stats" 
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        self.raw_log_var_graphs_dir = self.stats_dir / "raw_log_var_graphs"
        self.raw_log_var_graphs_dir.mkdir(parents=True, exist_ok=True)

        self.movies_dir = self.output_dir / "movies"
        self.movies_dir.mkdir(parents=True, exist_ok=True)

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

        writer = imageio.get_writer(self.movies_dir / 'vae_movie.mp4', fps=10)
        for curr_frame in tqdm(frames, desc='Processing frames'):
            im = imageio.imread(Path(curr_frame))
            writer.append_data(im)
        writer.close()

    def make_mu_log_var_graphs(self):
        with open(self.stats_dir / Path('mu_log_var_lists.pkl'),'rb') as f:
            mu, log_var = pickle.load(f)
  
        lower_lim = np.percentile(log_var, 5, axis=1).min()
        upper_lim = np.percentile(log_var, 95, axis=1).max()
        for ctr, data in enumerate(tqdm(log_var, desc="Processing log_var")):
            #sort_data=np.sort(np.exp(data.numpy()))
            sort_data=np.sort(data.numpy())
            fig, ax = plt.subplots()  # Create a figure containing a single axes.
            ax.set_xlabel('Ordered Indices')
            ax.set_ylabel('Log Var')
            ax.axis([0,512, lower_lim, upper_lim])
            ax.set_title('Sample'+ str(ctr))
            ax.plot(range(512), sort_data)
            plt_name = Path('log_var_' + str(ctr) + '.png')
            plt.savefig(self.raw_log_var_graphs_dir / plt_name)
            plt.close('all')

        def frame_num(in_path):
            frame_num = int(in_path.name.split('_')[2].split('.')[0])
            return frame_num
        
        plt_files = sorted([x for x in Path(self.raw_log_var_graphs_dir).iterdir() 
                            if x.name.startswith('log_var_')],
                            key=frame_num)

        writer = imageio.get_writer(self.movies_dir / 'log_var.mp4', fps=20)
        for file in tqdm(plt_files, desc='Making log_var movie'):
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
        axes[0].plot(range(len(recon_loss_list)), recon_loss_list, label="recon\n loss")
        axes[0].plot(range(len(recon_loss_list)), kl_loss_list, label="kl\n loss")
        axes[0].set_ylim([.1,1000])
        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})

        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Recon Loss')
        axes[1].plot(range(len(recon_loss_list)), recon_loss_list, label="recon\n loss")
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})

        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('KL Loss')
        axes[2].plot(range(len(recon_loss_list)), kl_loss_list, label="kl\n loss", color='#ff7f0e')
        axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})

        plt.savefig(self.stats_dir / Path('Recon_KL_Comp_1.png'))


    def compare_recon_kl_losses2(self):

        with open(self.stats_dir / Path('loss_lists.pkl'), 'rb') as f:
            recon_loss_list, kl_loss_list, adj_kl_factor_list = pickle.load(f)
        num_pts = len(recon_loss_list)

        # Create an EngFormatter object with desired precision
        formatter = EngFormatter(places=0, unit='')  # Adjust places and unit as needed


        fig, axes = plt.subplots(2,2)  # Create a figure containing a single axes.
        axes[0][0].set_xlabel('Iteration',fontsize=8, labelpad=-2)
        axes[0][0].set_ylabel('Loss')
        axes[0][0].set_yscale('log')
        axes[0][0].plot(range(num_pts), recon_loss_list, label="recon loss")
        axes[0][0].plot(range(num_pts), kl_loss_list, label="kl loss")
        axes[0][0].set_ylim([.0001,1000])

        axes[0][1].set_xlabel('Iteration',fontsize=8, labelpad=-2)
        axes[0][1].set_ylabel('Adj KL Loss')
        axes[0][1].yaxis.set_major_formatter(formatter)
        axes[0][1].plot(range(num_pts), adj_kl_factor_list, label="adj kl loss", color='#ff7f0e')
        axes[0][1].yaxis.tick_right()
        axes[0][1].yaxis.set_label_position("right")

        axes[1][0].set_xlabel('Iteration')
        axes[1][0].set_ylabel('Recon Loss')
        axes[1][0].plot(range(num_pts), recon_loss_list, label="recon loss")

        axes[1][1].set_xlabel('Iteration')
        axes[1][1].set_ylabel('KL Loss')
        axes[1][1].yaxis.tick_right()
        axes[1][1].yaxis.set_label_position("right")
        axes[1][1].plot(range(num_pts), kl_loss_list, label="kl loss", color='#ff7f0e')

        plt.savefig(self.stats_dir / Path('Recon_KL_Comp_2.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="scratch_output") 
    args = parser.parse_args()
    ar = AnalyzeResults(args.output_dir)
    ar.make_images_movie()
    ar.make_mu_log_var_graphs()
    ar.compare_recon_kl_losses()    
    ar.compare_recon_kl_losses2()