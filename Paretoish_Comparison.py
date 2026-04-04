import argparse
import matplotlib.pyplot as plt

from pathlib import Path

from VAE_Anime_ArtifactReader import ArtifactReader
from VAE_Anime_Config import TrainerConfig
from VAE_Anime_ResultsIO import read_loss_file_points_io
from VAE_ParetoFront import ParetoFront

from utils import get_experiment_dir


def get_experiment_label(expt_dir, fallback_label=None):
    expt_dir = Path(expt_dir)
    cfg_path = expt_dir / "config.ini"

    if cfg_path.is_file():
        cfg = TrainerConfig.from_file(cfg_path)
        if cfg.expt_name:
            return f"{expt_dir.name}: {cfg.expt_name}"
        
    if fallback_label is not None:
        return f"{expt_dir.name}: {fallback_label}"

    return expt_dir.name

def read_pareto_points(expt_dir):
    stats_dir = Path(expt_dir) / "stats"
    reader = ArtifactReader(stats_dir)

    try:
        series = reader.read_loss_series()
        return series.recon_loss, series.kl_loss
    except Exception:
        losses_file = resolve_losses_file(expt_dir)
        _, recon_pts, kl_pts, _ = read_loss_file_points_io(losses_file)
        return recon_pts, kl_pts

def resolve_experiment_dir(parent_dir, expt_num=None, expt_dir=None):
    if expt_dir is not None:
        resolved = Path(expt_dir)
    elif expt_num is not None:
        resolved = get_experiment_dir(parent_dir, expt_num)
    else:
        raise ValueError("Must provide either experiment number or experiment directory")

    if not resolved.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {resolved}")
    return resolved


def resolve_losses_file(expt_dir):
    losses_file = Path(expt_dir) / 'stats' / 'losses_file.txt'
    if not losses_file.is_file():
        raise FileNotFoundError(f"Loss file does not exist: {losses_file}")
    return losses_file


def compare_graphs(expt1_label, recon_pts1, kl_pts1, 
                   expt2_label, recon_pts2, kl_pts2,
                   output_dir):
    fig, ax = plt.subplots(2)

    # Early points destroy scaling, so skip 1st 10% of the points
    skip_pts1 = int(.10*len(recon_pts1))
    skip_pts2 = int(.10*len(recon_pts2))

    # create a scatter plot on the axes for expt1
    sc = ax[0].scatter(recon_pts1[skip_pts1:], kl_pts1[skip_pts1:],
                    s=1, c='cadetblue', label=f'Expt {expt1_label}')

    # create a scatter plot on the axes for expt2
    sc = ax[0].scatter(recon_pts2[skip_pts2:], kl_pts2[skip_pts2:],
                    s=1, c='indianred', label=f'Expt {expt2_label}')

    # Give the plot a title and labels
    ax[0].set_title('Pareto-ish Graph')
    ax[0].set_xlabel('Recon Loss')
    ax[0].set_ylabel('KL Loss')
    ax[0].set_yscale('log')
    ax[0].legend()

    # Now repeat in reverse order to avoid overlap confusion
    # create a scatter plot on the axes for expt2
    ax[1].scatter(recon_pts2[skip_pts2:], kl_pts2[skip_pts2:],
                    s=1, c='indianred', label=expt2_label)
    
    # create a scatter plot on the axes for expt1
    ax[1].scatter(recon_pts1[skip_pts1:], kl_pts1[skip_pts1:],
                    s=1, c='cadetblue', label=expt1_label)


    # Give the plot a title and labels
    ax[1].set_xlabel('Recon Loss')
    ax[1].set_ylabel('KL Loss')
    ax[1].set_yscale('log')
    ax[1].legend()

    plt.savefig(output_dir / Path(f'Paretoish {expt1_label}  {expt2_label}.png'))
    plt.close()
    print(f"Saved {output_dir / Path(f'Paretoish {expt1_label}  {expt2_label}.png')}")

def compare_pareto_curves(expt1_label, recon_pts1, kl_pts1, 
                          expt2_label, recon_pts2, kl_pts2,
                          output_dir):
    p1 = ParetoFront()
    p2 = ParetoFront()

    # Early points are noisy, so skip 1st 10% of the points
    skip_pts1 = int(.10*len(recon_pts1))
    skip_pts2 = int(.10*len(recon_pts2))


    p1.add_points(recon_pts1[skip_pts1:], kl_pts1[skip_pts1:])
    p2.add_points(recon_pts2[skip_pts2:], kl_pts2[skip_pts2:])

    pareto_curve1 = p1.get_smooth_pareto_curve()
    pareto_curve2 = p2.get_smooth_pareto_curve()

    fig, ax = plt.subplots()

    ax.plot(pareto_curve1[:, 0], pareto_curve1[:, 1], 
            color='dodgerblue', label=expt1_label, linewidth=2)
    ax.scatter(pareto_curve1[:, 0], pareto_curve1[:, 1], 
                   color='darkslategray', s=10)  # show curve points
    
    ax.plot(pareto_curve2[:, 0], pareto_curve2[:, 1], 
            color='lightcoral', label=expt2_label, linewidth=2)
    ax.scatter(pareto_curve2[:, 0], pareto_curve2[:, 1], 
                   color='maroon', s=10)  # show curve points
    
    ax.set_xlabel('Recon Loss')
    ax.set_ylabel('KL Loss')
    ax.set_yscale('log')
    ax.set_title('Pareto Curves')
    ax.legend()

    plt.savefig(output_dir / Path(f'ParetoCurves {expt1_label}  {expt2_label}.png'))
    plt.close()
    print(f"Saved {output_dir / Path(f'ParetoCurves {expt1_label}  {expt2_label}.png')}")


 


if __name__ == "__main__":
    '''Compare the pareto-ish graphs of two experiments.'''
    parser = argparse.ArgumentParser(description='Set some params for training & output dir.')
    parser.add_argument('--expt1', type=int, help='Number of first experiment')
    parser.add_argument('--expt2', type=int, help='Number of second experiment')
    parser.add_argument('--expt1_dir', type=str, help='Path to first experiment directory', default = None)
    parser.add_argument('--expt2_dir', type=str, help='Path to second experiment directory', default = None)
    parser.add_argument('--output_dir', type=str, help='Directory for comparison outputs', default = None)

    parser.add_argument('-c', '--config_file', type=str, nargs='?',
                        default='config.ini', help='Config file')

    args = parser.parse_args()

    # Load the experiment files 
    config_file = args.config_file
    cfg = TrainerConfig.from_file(config_file)
    parent_dir = cfg.parent_dir


    e1 = args.expt1
    e2 = args.expt2

    if args.expt1_dir:
        e1_dir_arg = args.expt1_dir
    else:
        e1_dir_arg = parent_dir / Path(f'expt_{e1}')

    if args.expt2_dir:
        e2_dir_arg = args.expt2_dir
    else:
        e2_dir_arg = parent_dir / Path(f'expt_{e2}')
        

    # Check if the experiment directories exist
    expt1_dir = resolve_experiment_dir(parent_dir, expt_num=e1, expt_dir=e1_dir_arg)
    expt2_dir = resolve_experiment_dir(parent_dir, expt_num=e2, expt_dir=e2_dir_arg)

    recon_pts1, kl_pts1 = read_pareto_points(expt1_dir)
    recon_pts2, kl_pts2 = read_pareto_points(expt2_dir)

    output_dir = Path(args.output_dir) if args.output_dir \
                                       else Path(parent_dir) / Path('pareto_comps')

    output_dir.mkdir(parents=True, exist_ok=True)

    expt1_label = get_experiment_label(expt1_dir, fallback_label=e1)
    expt2_label = get_experiment_label(expt2_dir, fallback_label=e2)

    compare_graphs(expt1_label, recon_pts1, kl_pts1, 
                   expt2_label, recon_pts2, kl_pts2,
                   output_dir)

    compare_pareto_curves(expt1_label, recon_pts1, kl_pts1, 
                          expt2_label, recon_pts2, kl_pts2,
                          output_dir)
