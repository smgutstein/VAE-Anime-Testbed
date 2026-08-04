import json

from ExptSummarizer import summarize_expt_artifacts
from VAE_Anime_Artifacts import LOSS_EVENTS_FILE


def _write_run_summary(expt_dir, status):
    (expt_dir / "run_summary.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )


def test_completed_run_summary_is_authoritative(tmp_path):
    expt_dir = tmp_path / "expt_1"
    stats_dir = expt_dir / "stats"
    stats_dir.mkdir(parents=True)
    (stats_dir / LOSS_EVENTS_FILE).touch()
    (stats_dir / "tripwire_skip_epoch1_step2.npz").touch()
    _write_run_summary(expt_dir, "completed")

    summary = summarize_expt_artifacts(expt_dir)

    assert summary == {
        "repo_commit": None,
        "repo_branch": None,
        "repo_dirty": None,
        "has_run_summary": True,
        "has_training_artifacts": True,
        "has_crash_artifacts": False,
        "has_stepguard_artifacts": True,
        "status": "completed",
    }


def test_failed_run_summary_overrides_artifact_fallback(tmp_path):
    expt_dir = tmp_path / "expt_2"
    stats_dir = expt_dir / "stats"
    stats_dir.mkdir(parents=True)
    (stats_dir / LOSS_EVENTS_FILE).touch()
    _write_run_summary(expt_dir, "failed")

    summary = summarize_expt_artifacts(expt_dir)

    assert summary["status"] == "failed"
    assert summary["has_crash_artifacts"] is False


def test_legacy_crash_artifact_is_classified_as_failed(tmp_path):
    expt_dir = tmp_path / "expt_3"
    stats_dir = expt_dir / "stats"
    stats_dir.mkdir(parents=True)
    (stats_dir / "precrash_history.npz").touch()

    summary = summarize_expt_artifacts(expt_dir)

    assert summary["has_run_summary"] is False
    assert summary["has_crash_artifacts"] is True
    assert summary["status"] == "failed"


def test_legacy_training_artifact_is_classified_as_artifacts_only(tmp_path):
    expt_dir = tmp_path / "expt_4"
    stats_dir = expt_dir / "stats"
    stats_dir.mkdir(parents=True)
    (stats_dir / LOSS_EVENTS_FILE).touch()

    summary = summarize_expt_artifacts(expt_dir)

    assert summary["has_training_artifacts"] is True
    assert summary["status"] == "artifacts_only"


def test_invalid_run_summary_produces_legacy_fallback(tmp_path):
    expt_dir = tmp_path / "expt_5"
    expt_dir.mkdir()
    (expt_dir / "run_summary.json").write_text("not-json", encoding="utf-8")

    summary = summarize_expt_artifacts(expt_dir)

    assert summary["has_run_summary"] is False
    assert summary["status"] == "config_only"


def test_repo_info_is_read_from_clean_notes(tmp_path):
    expt_dir = tmp_path / "expt_6"
    expt_dir.mkdir()
    (expt_dir / "Notes.txt").write_text(
        """Git Hash:
Current Branch: main

commit 476f51750ff9a0230a9a50722baae6bd7fe6ab36
Author: Test User <test@example.com>
""",
        encoding="utf-8",
    )

    summary = summarize_expt_artifacts(expt_dir)

    assert summary["repo_commit"] == "476f51750ff9a0230a9a50722baae6bd7fe6ab36"
    assert summary["repo_branch"] == "main"
    assert summary["repo_dirty"] is False


def test_repo_info_marks_training_code_changes_as_dirty(tmp_path):
    expt_dir = tmp_path / "expt_7"
    expt_dir.mkdir()
    (expt_dir / "Notes.txt").write_text(
        """Current Branch: feature/controller

commit 5bd0e270f210a23ca5aad1eb8535fda9cbd02226

Uncommitted changes:
diff --git a/VAE_Anime_Train.py b/VAE_Anime_Train.py
""",
        encoding="utf-8",
    )

    summary = summarize_expt_artifacts(expt_dir)

    assert summary["repo_commit"] == "5bd0e270f210a23ca5aad1eb8535fda9cbd02226"
    assert summary["repo_branch"] == "feature/controller"
    assert summary["repo_dirty"] is True


def test_missing_notes_produces_unknown_repo_info(tmp_path):
    expt_dir = tmp_path / "expt_8"
    expt_dir.mkdir()

    summary = summarize_expt_artifacts(expt_dir)

    assert summary["repo_commit"] is None
    assert summary["repo_branch"] is None
    assert summary["repo_dirty"] is None


def test_repo_info_ignores_config_only_changes(tmp_path):
    expt_dir = tmp_path / "expt_9"
    expt_dir.mkdir()
    (expt_dir / "Notes.txt").write_text(
        """Current Branch: main

commit 476f51750ff9a0230a9a50722baae6bd7fe6ab36

Uncommitted changes:
diff --git a/configs/config_beta_1.ini b/configs/config_beta_1.ini
index 1111111..2222222 100644
--- a/configs/config_beta_1.ini
+++ b/configs/config_beta_1.ini
""",
        encoding="utf-8",
    )

    summary = summarize_expt_artifacts(expt_dir)

    assert summary["repo_dirty"] is False


def test_repo_info_ignores_unrelated_python_changes(tmp_path):
    expt_dir = tmp_path / "expt_10"
    expt_dir.mkdir()
    (expt_dir / "Notes.txt").write_text(
        """Current Branch: main

commit 476f51750ff9a0230a9a50722baae6bd7fe6ab36

Uncommitted changes:
diff --git a/ExptSummarizer.py b/ExptSummarizer.py
index 1111111..2222222 100644
--- a/ExptSummarizer.py
+++ b/ExptSummarizer.py
""",
        encoding="utf-8",
    )

    summary = summarize_expt_artifacts(expt_dir)

    assert summary["repo_dirty"] is False
