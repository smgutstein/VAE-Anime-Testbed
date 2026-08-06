import configparser
import contextlib
import cProfile
import logging
import math
import shutil
import subprocess as sp
import time

from pathlib import Path

    
def _run_git_cmd(args, repo_dir=None, binary=False):
    """Run a git command and return its output, or None if git is unavailable."""
    try:
        output = sp.check_output(
            ["git", *args],
            cwd=repo_dir,
            stderr=sp.DEVNULL,
        )
    except Exception:
        return None

    if binary:
        return output

    return output.decode("utf-8", errors="surrogateescape").strip()


def _git_path_list(args, repo_dir):
    """Return a null-delimited git path list without breaking on spaces."""
    output = _run_git_cmd([*args, "-z"], repo_dir=repo_dir, binary=True)
    if output is None:
        return []

    return [
        Path(path.decode("utf-8", errors="surrogateescape"))
        for path in output.split(b"\0")
        if path
    ]


def _copy_worktree_files(repo_root, relative_paths, destination_root):
    for relative_path in relative_paths:
        source = repo_root / relative_path
        if not source.exists() and not source.is_symlink():
            # Deleted tracked files are represented by diff.patch.
            continue

        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)


def get_git_hash(repo_dir=None):
    """Return branch and latest commit information, if available."""
    inside_repo = _run_git_cmd(
        ["rev-parse", "--is-inside-work-tree"],
        repo_dir=repo_dir,
    )

    if inside_repo != "true":
        return "Git info unavailable (not a git repository)."

    branch = _run_git_cmd(["branch", "--show-current"], repo_dir=repo_dir) or "unknown"
    commit = _run_git_cmd(["log", "-n", "1"], repo_dir=repo_dir) or "unknown"

    return "\n".join([
        f"Current Branch: {branch}",
        "",
        commit,
    ])


def snapshot_source_state(experiment_dir, repo_dir=None):
    """
    Snapshot repository changes that are not represented by the current commit.

    The snapshot preserves three distinct states:
      - staged: index versions of staged tracked files + staged diff
      - unstaged: working-tree versions of unstaged tracked files + unstaged diff
      - untracked: exact copies of untracked, non-ignored files

    Returns concise text suitable for inclusion in Notes.txt.
    """
    experiment_dir = Path(experiment_dir).resolve()

    repo_root_str = _run_git_cmd(
        ["rev-parse", "--show-toplevel"],
        repo_dir=repo_dir,
    )
    if not repo_root_str:
        return "Repository state unavailable (not a git repository)."

    repo_root = Path(repo_root_str).resolve()
    snapshot_root = experiment_dir / "source_snapshot"

    staged_dir = snapshot_root / "staged"
    unstaged_dir = snapshot_root / "unstaged"
    untracked_dir = snapshot_root / "untracked"

    staged_files_dir = staged_dir / "files"
    unstaged_files_dir = unstaged_dir / "files"
    untracked_files_dir = untracked_dir / "files"

    for directory in (staged_files_dir, unstaged_files_dir, untracked_files_dir):
        directory.mkdir(parents=True, exist_ok=True)

    staged_patch = _run_git_cmd(
        ["diff", "--cached", "--binary"],
        repo_dir=repo_root,
    ) or ""
    unstaged_patch = _run_git_cmd(
        ["diff", "--binary"],
        repo_dir=repo_root,
    ) or ""

    (staged_dir / "diff.patch").write_text(
        staged_patch + ("\n" if staged_patch else ""),
        encoding="utf-8",
        errors="surrogateescape",
    )
    (unstaged_dir / "diff.patch").write_text(
        unstaged_patch + ("\n" if unstaged_patch else ""),
        encoding="utf-8",
        errors="surrogateescape",
    )

    staged_paths = _git_path_list(
        ["diff", "--cached", "--name-only"],
        repo_root,
    )
    staged_copy_paths = _git_path_list(
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        repo_root,
    )
    unstaged_paths = _git_path_list(
        ["diff", "--name-only"],
        repo_root,
    )
    untracked_paths = _git_path_list(
        ["ls-files", "--others", "--exclude-standard"],
        repo_root,
    )

    # If experiment output is inside the repository and is not ignored, do not
    # recursively snapshot the experiment directory that we are currently creating.
    try:
        experiment_relative = experiment_dir.relative_to(repo_root)
    except ValueError:
        experiment_relative = None

    if experiment_relative is not None:
        untracked_paths = [
            path for path in untracked_paths
            if path != experiment_relative and experiment_relative not in path.parents
        ]

    # A staged file must be copied from Git's index, not from the working tree.
    # This matters when a file has both staged and unstaged edits.
    for relative_path in staged_copy_paths:
        contents = _run_git_cmd(
            ["show", f":{relative_path.as_posix()}"],
            repo_dir=repo_root,
            binary=True,
        )
        if contents is None:
            continue

        destination = staged_files_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)

    _copy_worktree_files(repo_root, unstaged_paths, unstaged_files_dir)
    _copy_worktree_files(repo_root, untracked_paths, untracked_files_dir)

    lines = [
        "Repository state at experiment start:",
        "-------------------------------------",
        "",
        f"Staged tracked changes: {len(staged_paths)} files",
        "Snapshot: source_snapshot/staged/",
        "Diff:     source_snapshot/staged/diff.patch",
        "",
        f"Unstaged tracked changes: {len(unstaged_paths)} files",
        "Snapshot: source_snapshot/unstaged/",
        "Diff:     source_snapshot/unstaged/diff.patch",
        "",
        f"Untracked, non-ignored files: {len(untracked_paths)} files",
        "Snapshot: source_snapshot/untracked/",
    ]

    for heading, paths in (
        ("Staged files:", staged_paths),
        ("Unstaged files:", unstaged_paths),
        ("Untracked files:", untracked_paths),
    ):
        lines.extend(["", heading])
        if paths:
            lines.extend(f"    {path.as_posix()}" for path in paths)
        else:
            lines.append("    (none)")

    return "\n".join(lines)

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
       in how I would update the KL weight. The idea is to
       have a function that can be customized to increase or
       decrease the KL weight in a variety of ways.'''

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
            "epochs", "learning_rate", "loss_policy", 
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

    training = "Training_Parameters"
    loss_policy = config.get(training, "loss_policy")

    if loss_policy not in {"adaptive_kl", "fixed_beta"}:
        raise ValueError(
            "Training_Parameters.loss_policy must be one of: adaptive_kl, fixed_beta"
        )
    
    if loss_policy == "fixed_beta":
        if not config.has_option(training, "beta"):
            raise ValueError(
                "Missing required config option: [Training_Parameters] beta"
            )


    if loss_policy == "adaptive_kl":
        if not config.has_option(training, "running_window"):
            raise ValueError(
                "Missing required config option: [Training_Parameters] running_window"
            )

        has_new_kl_keys = all(
            config.has_option(training, opt)
            for opt in (
                "initial_kl_weight",
                "max_kl_weight",
                "kl_weight_update_factor",
            )
        )
        has_old_kl_keys = all(
            config.has_option(training, opt)
            for opt in (
                "kl_adj_factor",
                "kl_adj_factor_max",
                "kl_adj_update_factor",
            )
        )

        if not (has_new_kl_keys or has_old_kl_keys):
            raise ValueError(
                "Missing KL-weight config options: require either "
                "[Training_Parameters] initial_kl_weight/max_kl_weight/kl_weight_update_factor "
                "or legacy kl_adj_factor/kl_adj_factor_max/kl_adj_update_factor"
            )

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