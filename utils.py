import configparser
import contextlib
import cProfile
import logging
import math
import numpy as np
import os
import random
import subprocess as sp
import tensorflow as tf
import time

from pathlib import Path

def set_all_seeds(seed: int, deterministic: bool = False):
    """Set Python, NumPy, and TensorFlow seeds.

    Args:
        seed: Integer seed value.
        deterministic: If True, request more deterministic TF behavior.
            This can reduce performance and is not guaranteed to make
            every GPU op bitwise identical.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as e:
            logging.warning(f"Could not enable TF op determinism: {e}")

def read_config_file(filename):
    '''Reads a configuration file and returns a ConfigParser object.'''
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def is_config_file(filename):
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Check if the line resembles an INI section header
                if line.strip().startswith('[') and line.strip().endswith(']'):
                    return True
                # Check if the line resembles a key-value pair
                if '=' in line:
                    return True
        return False
    except FileNotFoundError:
        return False
    
def get_git_hash():
    '''Returns git info if available; otherwise returns a safe fallback string.'''

    def run_git_cmd(cmd):
        try:
            return sp.check_output(cmd, stderr=sp.DEVNULL).decode("utf-8").strip()
        except Exception:
            return None

    # Check if we're inside a git repo
    inside_repo = run_git_cmd(['git', 'rev-parse', '--is-inside-work-tree'])

    if inside_repo != 'true':
        return "Git info unavailable (not a git repository)."

    branch = run_git_cmd(['git', 'branch', '--show-current']) or "unknown"
    commit = run_git_cmd(['git', 'log', '-n', '1']) or "unknown"
    diff = run_git_cmd(['git', 'diff']) or ""

    output = []
    output.append(f"Current Branch: {branch}")
    output.append("")
    output.append(commit)

    if diff:
        output.append("")
        output.append("Uncommitted changes:")
        output.append(diff)

    return "\n".join(output)

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper


def profile_decorator(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f'profiler_output_{timestamp}.txt', 'w') as f:
            with contextlib.redirect_stdout(f):
                profiler.print_stats(sort='cumtime')

        return result

    return wrapper

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# Retained intentionally: alternate KL-adjustment update modes
# (multiplicative vs additive) are still under evaluation.
def delt_add(x, y):
    return x + y    
def delt_sub(x,y):
    return x - y    
def delt_mul(x,y):       
    return x * y
def delt_div(x,y):
    return x / y
    
class DeltaGenerator():
    '''Overly generalized function used to give more flexibility
       in how I would update the kl_adj_factor. The idea is to  
       have a function that can be customized to increase or
       decrease the kl_adj_factor in a variety of ways.'''
    def __init__(self, delta_inc_func, delta_dec_func, delta):

        self.delta = delta
        self.inc_func = self.customize_function(delta_inc_func, 1 + delta)
        self.dec_func = self.customize_function(delta_dec_func, 1 + delta)

    def customize_function(self, base_function, y):
        def custom_function(x):
            return base_function(x, y)
        return custom_function
    
# Retained intentionally: used for debugging runs where VAE loss
# trajectories suddenly diverge to NaN/Inf.
def find_nan_or_inf_index(lst):
    # Returns value of first NaN or inf in a list, so list can be truncated threre
    # If non nan or inf, list doesn't need truncation, so list length is returned
    try:
        # Find the index of the first element that satisfies the condition
        index = next(i for i, value in enumerate(lst) if math.isnan(value) or math.isinf(value))
        return index
    except StopIteration:
        # If no NaN or inf is found, raise an IndexError or return None
        return len(lst)
    
def setup_logging(level="INFO"):
    # Define accepted level aliases
    aliases = {
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    level_upper = level.upper()
    numeric_level = aliases.get(level_upper)

    if numeric_level is None:
        print(f"Invalid log level: '{level}'")
        print(f"Valid options: {', '.join(aliases.keys())}")
        return  # Or fall back to a default: numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s (%(levelname)s) : %(message)s"
    )

########################################
# Boolean parsing
########################################
def parse_bool(value, field_name="value"):
    """
    Robust boolean parser.
    Accepts: true/false, 1/0, yes/no (case-insensitive)
    """
    if isinstance(value, bool):
        return value

    if value is None:
        raise ValueError(f"Missing boolean value for '{field_name}'")

    val = str(value).strip().lower()

    if val in {"true", "1", "yes", "y"}:
        return True
    elif val in {"false", "0", "no", "n"}:
        return False
    else:
        raise ValueError(f"Invalid boolean for '{field_name}': {value}")


########################################
# Config loading + validation
########################################
def load_and_validate_config(config_file):
    """
    Load config and enforce required structure.
    """
    if not Path(config_file).exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config = configparser.ConfigParser()
    config.read(config_file)

    required_sections = [
        "Training_Parameters",
        "Output_Parameters",
        "Data_Parameters",
        "Model_Parameters",
        "Monitoring_Parameters",
    ]

    for section in required_sections:
        if not config.has_section(section):
            raise ValueError(f"Missing required config section: [{section}]")

    # Optional sections
    if not config.has_section("Reproducibility"):
        logging.warning("Config missing [Reproducibility]; using defaults")

    required_options = {
        "Training_Parameters": [
            "epochs", "learning_rate",  "loss_policy",
            "kl_adj_factor", "kl_adj_factor_max",
            "kl_adj_update_factor", "running_window",
        ],
        "Output_Parameters": ["parent_dir", "save_net"],
        "Data_Parameters": [
            "data_dir", "batch_size", "image_size", "val_split",
            "shuffle_buffer", "train_drop_remainder",
        ],
        "Model_Parameters": [
            "latent_dim", "base_filters", "filter_factors",
            "encode_dense_units", "kernel_size",
        ],
        "Monitoring_Parameters": [
            "snapshot_every", "train_preview_count", "valid_preview_count",
            "take_initial_snapshot", "run_analysis", "make_mu_log_var_movies",
        ],
    }

    for section, options in required_options.items():
        for option in options:
            if not config.has_option(section, option):
                raise ValueError(f"Missing required config option: [{section}] {option}")

    return config


########################################
# Config access helpers
########################################
def get_parent_dir(config):
    return Path(config.get("Output_Parameters", "parent_dir"))


def get_data_dir(config):
    if config.has_section("Data_Parameters") and config.has_option("Data_Parameters", "data_dir"):
        return Path(config.get("Data_Parameters", "data_dir"))
    else:
        # fallback default
        return Path("./data/anime")


def get_seed_and_determinism(config):
    """
    Backward-compatible:
    - Prefer [Reproducibility]
    - Fallback to [Training_Parameters] (deprecated)
    """
    seed = 1234
    deterministic = False

    if config.has_section("Reproducibility"):
        if config.has_option("Reproducibility", "seed"):
            seed = config.getint("Reproducibility", "seed")
        if config.has_option("Reproducibility", "deterministic"):
            deterministic = parse_bool(
                config.get("Reproducibility", "deterministic"),
                "deterministic"
            )

    elif config.has_section("Training_Parameters"):
        # backward compatibility
        if config.has_option("Training_Parameters", "seed"):
            seed = config.getint("Training_Parameters", "seed")
            logging.warning("Using deprecated location for 'seed' in [Training_Parameters]")

        if config.has_option("Training_Parameters", "deterministic"):
            deterministic = parse_bool(
                config.get("Training_Parameters", "deterministic"),
                "deterministic"
            )
            logging.warning("Using deprecated location for 'deterministic'")

    return seed, deterministic


########################################
# Experiment directory helpers
########################################
def list_experiment_dirs(parent_dir):
    """
    Returns sorted list of experiment directories (expt_N)
    """
    parent_dir = Path(parent_dir)

    if not parent_dir.exists():
        return []

    expt_dirs = []
    for d in parent_dir.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith("expt_"):
            continue

        suffix = d.name[len("expt_"):]
        if suffix.isdigit():
            expt_dirs.append(d)
        else:
            logging.warning(f"Skipping malformed experiment directory name: {d.name}")

    def extract_num(d):
        try:
            return int(d.name.split("_")[-1])
        except Exception:
            return -1

    expt_dirs.sort(key=extract_num)
    return expt_dirs


def get_latest_experiment_dir(parent_dir):
    expt_dirs = list_experiment_dirs(parent_dir)

    if not expt_dirs:
        raise RuntimeError(f"No experiment directories found in {parent_dir}")

    return expt_dirs[-1]


def get_next_experiment_dir(parent_dir):
    """
    Returns (next_expt_num, path)
    """
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    expt_dirs = list_experiment_dirs(parent_dir)

    if not expt_dirs:
        next_num = 1
    else:
        last = expt_dirs[-1].name
        last_num = int(last.split("_")[-1])
        next_num = last_num + 1

    return next_num, parent_dir / f"expt_{next_num}"


def get_experiment_dir(parent_dir, expt_num):
    """
    Returns path to specific experiment.
    """
    path = Path(parent_dir) / f"expt_{expt_num}"

    if not path.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {path}")

    return path