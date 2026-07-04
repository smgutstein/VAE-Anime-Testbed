from __future__ import annotations

import re
import argparse
import configparser
from pathlib import Path

import pandas as pd


EXPT_RE = re.compile(r"expt_(\d+)$")


CRASH_PATTERNS = (
    "*crash*",
    "*failed*",
)

STEPGUARD_PATTERNS = (
    "*stepguard*",
    "*step_guard*",
    "*toxic*",
)


def _has_matching_file(expt_dir: Path, patterns: tuple[str, ...]) -> bool:
    """
    Return True if any file below expt_dir matches one of the supplied glob patterns.
    """
    for pattern in patterns:
        if any(path.is_file() for path in expt_dir.rglob(pattern)):
            return True
    return False


def _has_history_file(expt_dir: Path) -> bool:
    """
    Return True if the experiment appears to contain a training-history artifact.

    This intentionally uses broad names so older experiment folders can still be
    summarized without requiring an exact artifact schema.
    """
    history_patterns = (
        "*history*.csv",
        "*history*.json",
        "*metrics*.csv",
        "*metrics*.json",
    )
    return _has_matching_file(expt_dir, history_patterns)


def _classify_expt_status(
    *,
    has_history: bool,
    has_crash_artifact: bool,
    has_stepguard_artifact: bool,
) -> str:
    """
    Classify experiment status from available artifacts.

    This is deliberately conservative. It detects obvious signs of crashes or
    StepGuard activity, but it does not pretend to know whether a run completed
    successfully unless richer run-summary metadata exists.
    """
    if has_crash_artifact:
        return "crashed"
    if has_stepguard_artifact:
        return "guarded"
    if has_history:
        return "has_history"
    return "config_only"


def summarize_expt_artifacts(expt_dir: Path) -> dict:
    """
    Summarize basic outcome/status artifacts for one experiment directory.
    """
    has_history = _has_history_file(expt_dir)
    has_crash_artifact = _has_matching_file(expt_dir, CRASH_PATTERNS)
    has_stepguard_artifact = _has_matching_file(expt_dir, STEPGUARD_PATTERNS)

    return {
        "has_history": has_history,
        "has_crash_artifact": has_crash_artifact,
        "has_stepguard_artifact": has_stepguard_artifact,
        "status": _classify_expt_status(
            has_history=has_history,
            has_crash_artifact=has_crash_artifact,
            has_stepguard_artifact=has_stepguard_artifact,
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
                "expt_nums",
                "num_expts",
            ]
        )
    summary = (
        df.groupby(["learning_rate", "loss_policy", "epochs", "beta", "status"], 
                   dropna=False)
        .agg(
            expt_nums=("expt_num", list),
            num_expts=("expt_num", "count"),
            has_history=("has_history", "sum"),
            has_crash_artifact=("has_crash_artifact", "sum"),
            has_stepguard_artifact=("has_stepguard_artifact", "sum"),
        )
        .reset_index()
        .sort_values(["learning_rate", "loss_policy", "epochs", "beta", "status"])
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
