from dataclasses import dataclass
from pathlib import Path
import logging
import shutil

from utils import get_git_hash
from utils import get_next_experiment_dir
from utils import snapshot_source_state


@dataclass(frozen=True)
class ExperimentRun:
    expt_num: int
    output_dir: Path
    raw_image_dir: Path
    stats_dir: Path
    movies_dir: Path
    model_info_dir: Path

    @classmethod
    def create(cls, cfg):
        expt_num, output_dir = get_next_experiment_dir(cfg.parent_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(str(cfg.config_file), output_dir / "config.ini")
        logging.info(f"Storing Expt {expt_num} in {output_dir}")

        with open(output_dir / "Notes.txt", "w") as f:
            hash_str = get_git_hash()
            f.write("Git Hash: \n")
            f.write(hash_str)
            f.write("\n\n")
            f.write(snapshot_source_state(output_dir))
            f.write("\n\n")
            f.write(f"Loss Policy: {cfg.loss_policy}\n")
            f.write(f"Random Seed: {cfg.seed}\n")
            f.write(f"Deterministic TF Ops: {cfg.deterministic}\n")

        raw_image_dir = output_dir / "raw_images"
        raw_image_dir.mkdir(parents=True, exist_ok=True)

        stats_dir = output_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        movies_dir = output_dir / "movies"
        movies_dir.mkdir(parents=True, exist_ok=True)

        model_info_dir = output_dir / "model_info"
        model_info_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            expt_num=expt_num,
            output_dir=output_dir,
            raw_image_dir=raw_image_dir,
            stats_dir=stats_dir,
            movies_dir=movies_dir,
            model_info_dir=model_info_dir,
        )