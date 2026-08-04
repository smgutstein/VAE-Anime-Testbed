from __future__ import annotations

import re
import json
import argparse
import configparser
from pathlib import Path

import pandas as pd

from VAE_Anime_Artifacts import (
    LATENT_KL_STATS_FILE,
    LATENT_STATS_FILE,
    LATENT_VAR_STATS_FILE,
    LOSS_EVENTS_FILE,
    LOSS_TEXT_FILE,
)


EXPT_RE = re.compile(r"expt_(\d+)$")


RUN_SUMMARY_FILE = "run_summary.json"
NOTES_FILE = "Notes.txt"
STATS_DIR = "stats"

BRANCH_RE = re.compile(r"^Current Branch:\s*(.+?)\s*$", re.MULTILINE)
COMMIT_RE = re.compile(r"^commit\s+([0-9a-fA-F]{7,40})\s*$", re.MULTILINE)
UNCOMMITTED_CHANGES_RE = re.compile(r"^Uncommitted changes:\s*$", re.MULTILINE)
DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)

# Files imported by, or executed as part of, VAE_Anime_Train.py. Changes to
# unrelated analysis scripts, tests, documentation, or experiment config files
# do not make the training code dirty.
EXPERIMENT_CODE_FILES = frozenset(
    {
        "VAE_Anime_Analysis.py",
        "VAE_Anime_ArtifactReader.py",
        "VAE_Anime_ArtifactWriter.py",
        "VAE_Anime_Artifacts.py",
        "VAE_Anime_Config.py",
        "VAE_Anime_Datasets.py",
        "VAE_Anime_Decoder.py",
        "VAE_Anime_Encoder.py",
        "VAE_Anime_ExperimentRun.py",
        "VAE_Anime_Full_Model.py",
        "VAE_Anime_KL_Weight_Scheduler.py",
        "VAE_Anime_LatentStatsPlotter.py",
        "VAE_Anime_LossPlotter.py",
        "VAE_Anime_LossPolicy.py",
        "VAE_Anime_MovieBuilder.py",
        "VAE_Anime_ResultsIO.py",
        "VAE_Anime_RunArtifacts.py",
        "VAE_Anime_Snapshotter.py",
        "VAE_Anime_StepGuard.py",
        "VAE_Anime_Train.py",
        "VAE_Anime_Training_Monitor.py",
        "VAE_Anime_TrainStep.py",
        "VAE_ParetoFront.py",
        "utils.py",
    }
)

TRAINING_ARTIFACT_FILES = (
    LOSS_TEXT_FILE,
    LOSS_EVENTS_FILE,
    LATENT_STATS_FILE,
    LATENT_VAR_STATS_FILE,
    LATENT_KL_STATS_FILE,
)

CRASH_PATTERNS = (
    "crash_step_epoch*_step*.npz",
    "last_good_before_crash_epoch*_step*.npz",
    "precrash_history.npz",
)

STEPGUARD_PATTERNS = (
    "tripwire_skip_epoch*_step*.npz",
)


def _has_matching_file(expt_dir: Path, patterns: tuple[str, ...]) -> bool:
    """
    Return True if any file below expt_dir matches one of the supplied glob patterns.
    """
    for pattern in patterns:
        if any(path.is_file() for path in expt_dir.rglob(pattern)):
            return True
    return False


def _read_run_summary(expt_dir: Path) -> dict | None:
    """
    Read the experiment's authoritative run summary.

    Return None when run_summary.json is absent, unreadable, malformed, or does
    not contain a JSON object.
    """
    summary_path = expt_dir / RUN_SUMMARY_FILE

    if not summary_path.is_file():
        return None

    try:
        with summary_path.open("r", encoding="utf-8") as summary_fh:
            summary = json.load(summary_fh)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(summary, dict):
        return None

    return summary


def _has_dirty_experiment_code(notes: str) -> bool:
    """Return True when Notes.txt records changes to training/runtime code."""
    if not UNCOMMITTED_CHANGES_RE.search(notes):
        return False

    changed_paths = {
        new_path
        for _, new_path in DIFF_FILE_RE.findall(notes)
    }
    return bool(changed_paths & EXPERIMENT_CODE_FILES)


def _read_repo_info(expt_dir: Path) -> dict:
    """
    Read repository metadata recorded in Notes.txt.

    Missing or unreadable notes produce unknown values. A repository is marked
    dirty only when the recorded Git diff changes code used by the experiment.
    Configuration, documentation, tests, and unrelated analysis scripts do not
    make an experiment's code state dirty.
    """
    notes_path = expt_dir / NOTES_FILE

    try:
        notes = notes_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {
            "repo_commit": None,
            "repo_branch": None,
            "repo_dirty": None,
        }

    branch_match = BRANCH_RE.search(notes)
    commit_match = COMMIT_RE.search(notes)

    return {
        "repo_commit": commit_match.group(1)[:6] if commit_match else None,
        "repo_branch": branch_match.group(1) if branch_match else None,
        "repo_dirty": _has_dirty_experiment_code(notes),
    }


def _has_training_artifacts(expt_dir: Path) -> bool:
    """
    Return True if the experiment contains a normal training-output artifact.
    """
    stats_dir = expt_dir / STATS_DIR
    return any((stats_dir / filename).is_file() for filename in TRAINING_ARTIFACT_FILES)


def _classify_expt_status(
    *,
    run_summary: dict | None,
    has_training_artifacts: bool,
    has_crash_artifacts: bool,
) -> str:
    """
    Determine experiment status from current metadata or legacy artifacts.

    run_summary.json is authoritative for current experiments. Artifact-based
    inference is retained for older experiment directories.
    """
    if run_summary is not None:
        status = run_summary.get("status")
        if status in {"completed", "failed"}:
            return status
        return "unknown"

    if has_crash_artifacts:
        return "failed"
    if has_training_artifacts:
        return "artifacts_only"
    return "config_only"


def summarize_expt_artifacts(expt_dir: Path) -> dict:
    """
    Summarize run status and artifact availability for one experiment.
    """
    run_summary = _read_run_summary(expt_dir)
    repo_info = _read_repo_info(expt_dir)
    has_training_artifacts = _has_training_artifacts(expt_dir)
    has_crash_artifacts = _has_matching_file(expt_dir, CRASH_PATTERNS)
    has_stepguard_artifacts = _has_matching_file(expt_dir, STEPGUARD_PATTERNS)

    return {
        **repo_info,
        "has_run_summary": run_summary is not None,
        "has_training_artifacts": has_training_artifacts,
        "has_crash_artifacts": has_crash_artifacts,
        "has_stepguard_artifacts": has_stepguard_artifacts,
        "status": _classify_expt_status(
            run_summary=run_summary,
            has_training_artifacts=has_training_artifacts,
            has_crash_artifacts=has_crash_artifacts,
        ),
    }


def _extract_expt_num(expt_dir: Path) -> int | None:
    """
    Extract numeric experiment id from folder names like expt_001, expt_23, etc.
    """
    match = EXPT_RE.match(expt_dir.name)
    if match is None:
        return None
    return int(match.group(1))


def read_expt_configs(expts_dir: str | Path = "./expts") -> pd.DataFrame:
    """
    Read every config.ini under expts_dir/expt_*/config.ini.

    Returns one row per experiment.
    """
    expts_dir = Path(expts_dir)

    rows = []

    for expt_dir in sorted(expts_dir.glob("expt_*")):
        if not expt_dir.is_dir():
            continue

        expt_num = _extract_expt_num(expt_dir)
        if expt_num is None:
            continue

        config_path = expt_dir / "config.ini"
        if not config_path.exists():
            continue

        config = configparser.ConfigParser()
        config.read(config_path)

        training = config["Training_Parameters"]

        rows.append(
            {
                "expt_num": expt_num,
                "expt_dir": str(expt_dir),
                "learning_rate": training.getfloat("learning_rate"),
                "loss_policy": training.get("loss_policy"),
                "epochs": training.getint("epochs"),
                "beta": (
                    training.getfloat("beta")
                    if training.get("loss_policy") == "fixed_beta"
                    else None
                ),
                **summarize_expt_artifacts(expt_dir),
            }
        )
    return pd.DataFrame(rows).sort_values("expt_num").reset_index(drop=True)


def summarize_expt_configs(expts_dir: str | Path = "./expts") -> pd.DataFrame:
    """
    Summarize experiments by learning_rate, loss_policy, and epochs.

    Returns one row per unique combination, with the matching expt numbers listed.
    """
    df = read_expt_configs(expts_dir)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "learning_rate",
                "loss_policy",
                "epochs",
                "beta",
                "status",
                "repo_commit",
                "repo_branch",
                "repo_dirty",
                "expt_nums",
                "num_expts",
            ]
        )
    summary = (
        df.groupby(
            [
                "learning_rate",
                "loss_policy",
                "epochs",
                "beta",
                "status",
                "repo_commit",
                "repo_branch",
                "repo_dirty",
            ],
            dropna=False,
        )
        .agg(
            expt_nums=("expt_num", list),
            num_expts=("expt_num", "count"),
            num_with_run_summary=("has_run_summary", "sum"),
            num_with_training_artifacts=("has_training_artifacts", "sum"),
            num_with_crash_artifacts=("has_crash_artifacts", "sum"),
            num_with_stepguard_artifacts=("has_stepguard_artifacts", "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "learning_rate",
                "loss_policy",
                "epochs",
                "beta",
                "status",
                "repo_commit",
            ]
        )
        .reset_index(drop=True)
    )
    return summary




def summarize_selected_expts(
    expt_nums: int | list[int] | tuple[int, ...] | set[int],
    expts_dir: str | Path = "./expts",
) -> pd.DataFrame:
    """
    Return one config/artifact-summary row per requested experiment.

    This is intended for notebook use:

        summarize_selected_expts([1, 2, 7])

    In Jupyter, the returned DataFrame will display naturally as a pandas table.
    """
    if isinstance(expt_nums, int):
        requested = [expt_nums]
    else:
        requested = list(expt_nums)

    df = read_expt_configs(expts_dir)

    if df.empty:
        return df

    return (
        df[df["expt_num"].isin(requested)]
        .sort_values("expt_num")
        .reset_index(drop=True)
    )


def summarize_expts(
    expts_dir: str | Path = "./expts",
    show_expts: int | list[int] | tuple[int, ...] | set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Return the grouped experiment summary and, optionally, selected experiments.

    This is useful in notebooks when you want both tables:

        summary, selected = summarize_expts(show_expts=[1, 2, 7])
        summary
        selected
    """
    summary = summarize_expt_configs(expts_dir)

    selected = None
    if show_expts is not None:
        selected = summarize_selected_expts(show_expts, expts_dir=expts_dir)

    return summary, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize VAE experiment configs and optionally show selected "
            "per-experiment config summaries."
        )
    )

    parser.add_argument(
        "--expts-dir",
        default="./expts",
        help="Directory containing expt_*/config.ini folders. Default: ./expts",
    )

    parser.add_argument(
        "--show-expts",
        nargs="+",
        type=int,
        default=None,
        metavar="EXPT_NUM",
        help=(
            "Optional experiment numbers to display in detail, in addition to "
            "the grouped summary table. Example: --show-expts 1 2 7"
        ),
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    pd.set_option("display.max_colwidth", None)

    summary, selected = summarize_expts(
        expts_dir=args.expts_dir,
        show_expts=args.show_expts,
    )

    print("\nGrouped experiment summary:")
    print(summary.to_string(index=False))

    if selected is not None:
        print("\nSelected experiment summaries:")

        if selected.empty:
            print(f"No matching experiments found for: {args.show_expts}")
        else:
            print(selected.to_string(index=False))

            found = set(selected["expt_num"].tolist())
            missing = sorted(set(args.show_expts) - found)
            if missing:
                print(
                    "\nWarning: no matching config.ini found "
                    f"for experiments: {missing}"
                )
