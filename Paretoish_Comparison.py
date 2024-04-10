import argparse
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import EngFormatter
from pathlib import Path
from tqdm import tqdm
from utils import is_config_file
from utils import read_config_file


def get_datapts(expt_file):
    with open(expt_file,'r') as f:
        fl = f.readlines()

    recon_pts=[]
    kl_pts = []
    for curr_line in fl[1:]:
        data = [x.strip() for x in curr_line.split('--')]
        recon_pts.append(float(data[2]))
        kl_pts.append(float(data[3]))

    return recon_pts, kl_pts

def compare_graphs(expt1_num, recon_pts1, kl_pts1, 
                   expt2_num, recon_pts2, kl_pts2,
                   output_dir):
    fig, ax = plt.subplots(2)

    # Early points destroy scaling, so skip 1st 10% of the points
    skip_pts1 = int(.10*len(recon_pts1))
    skip_pts2 = int(.10*len(recon_pts2))

    # create a scatter plot on the axes for expt1
    sc = ax[0].scatter(recon_pts1[skip_pts1:], kl_pts1[skip_pts1:],
                    s=1, c='cadetblue', label=f'Expt {expt1_num}')

    # create a scatter plot on the axes for expt2
    sc = ax[0].scatter(recon_pts2[skip_pts2:], kl_pts2[skip_pts2:],
                    s=1, c='indianred', label=f'Expt {expt2_num}')

    # Give the plot a title and labels
    ax[0].set_title('Pareto-ish Graph')
    ax[0].set_xlabel('Recon Loss')
    ax[0].set_ylabel('KL Loss')
    ax[0].set_yscale('log')
    ax[0].legend()

    # Now repeat in reverse order to avoid overlap confusion
    # create a scatter plot on the axes for expt2
    sc = ax[1].scatter(recon_pts2[skip_pts2:], kl_pts2[skip_pts2:],
                    s=1, c='indianred', label=f'Expt {expt2_num}')
    
    # create a scatter plot on the axes for expt1
    sc = ax[1].scatter(recon_pts1[skip_pts1:], kl_pts1[skip_pts1:],
                    s=1, c='cadetblue', label=f'Expt {expt1_num}')


    # Give the plot a title and labels
    ax[1].set_xlabel('Recon Loss')
    ax[1].set_ylabel('KL Loss')
    ax[1].set_yscale('log')
    ax[1].legend()

    plt.savefig(output_dir / Path(f'Paretoish_{expt1_num}_{expt2_num}.png'))
    plt.close()
    print(f"Saved {output_dir / Path(f'Paretoish_{expt1_num}_{expt2_num}.png')}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set some params for training & output dir.')
    parser.add_argument('--expt1', type=int, help='Number of first experiment file')    
    parser.add_argument('--expt2', type=int, help='Number of second experiment file')
    parser.add_argument('-c', '--config_file', type=str, nargs='?',
                        default='config.ini', help='Config file')

    args = parser.parse_args()
    e1 = args.expt1
    e2 = args.expt2
    config_file = args.config_file

    # Load the experiment files 
    assert is_config_file(config_file), "Invalid config file"
    config = read_config_file(config_file)
    parent_dir = config.get('Output_Parameters', 'parent_dir')

    # Check if the experiment directories exist
    expt1_dir = parent_dir / Path(f'expt_{e1}') 
    expt2_dir = parent_dir / Path(f'expt_{e2}')  
    assert expt1_dir.is_dir(), f"Experiment {e1} directory - {expt1_dir}, does not exist"
    assert expt2_dir.is_dir(), f"Experiment {e2} directory - {expt2_dir}, does not exist"

    # Check if the experiment files exist
    expt1_file = expt1_dir / Path('stats/losses_file.txt')
    expt2_file = expt2_dir / Path('stats/losses_file.txt')
    assert expt1_file.is_file(), f"Experiment {e1} file - {expt1_file}, does not exist"
    assert expt2_file.is_file(), f"Experiment {e2} file - {expt2_file}, does not exist"

    # Get the data points
    recon_pts1, kl_pts1 = get_datapts(expt1_file)
    recon_pts2, kl_pts2 = get_datapts(expt2_file)

    output_dir = Path(parent_dir) / Path('pareto_comps')
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_graphs(e1, recon_pts1, kl_pts1, 
                   e2, recon_pts2, kl_pts2,
                   output_dir)

