import subprocess

import pytest


class TestParseBool:
    @pytest.fixture(autouse=True)
    def _import(self):
        from utils import parse_bool
        self.parse_bool = parse_bool

    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "Yes", "y", "Y"])
    def test_truthy_values(self, val):
        assert self.parse_bool(val, "field") is True

    @pytest.mark.parametrize("val", ["false", "False", "FALSE", "0", "no", "No", "n", "N"])
    def test_falsy_values(self, val):
        assert self.parse_bool(val, "field") is False

    def test_invalid_value_raises_naming_the_field(self):
        with pytest.raises(ValueError, match="my_field"):
            self.parse_bool("maybe", "my_field")


class TestDeltaGenerator:
    @pytest.fixture(autouse=True)
    def _import(self):
        from utils import DeltaGenerator, delt_mul, delt_div
        self.DG = DeltaGenerator
        self.mul = delt_mul
        self.div = delt_div

    def test_multiplicative_increase(self):
        dg = self.DG(self.mul, self.div, delta=0.1)
        assert dg.inc_func(1.0) == pytest.approx(1.1)

    def test_inc_then_dec_is_near_identity(self):
        dg = self.DG(self.mul, self.div, delta=0.1)
        assert dg.dec_func(dg.inc_func(1.0)) == pytest.approx(1.0, rel=1e-6)


class TestExperimentDirHelpers:
    @pytest.fixture(autouse=True)
    def _import(self):
        from utils import list_experiment_dirs, get_next_experiment_dir, get_latest_experiment_dir, get_experiment_dir
        self.list_dirs = list_experiment_dirs
        self.get_next = get_next_experiment_dir
        self.get_latest = get_latest_experiment_dir
        self.get_dir = get_experiment_dir

    def test_sorted_numerically_not_lexicographically(self, tmp_path):
        for n in [1, 2, 9, 10, 11]:
            (tmp_path / f"expt_{n}").mkdir()
        nums = [int(d.name.split("_")[1]) for d in self.list_dirs(tmp_path)]
        assert nums == [1, 2, 9, 10, 11]

    def test_get_next_goes_one_past_highest_even_with_gaps(self, tmp_path):
        (tmp_path / "expt_1").mkdir()
        (tmp_path / "expt_5").mkdir()
        num, path = self.get_next(tmp_path)
        assert num == 6
        assert path.name == "expt_6"

    def test_get_latest_on_empty_parent_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            self.get_latest(tmp_path)

    def test_get_experiment_dir_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self.get_dir(tmp_path, 99)


class TestSourceSnapshot:
    @staticmethod
    def _git(repo, *args):
        subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_snapshot_preserves_staged_unstaged_and_untracked_states(self, tmp_path):
        from utils import snapshot_source_state

        repo = tmp_path / "repo"
        repo.mkdir()

        self._git(repo, "init")
        self._git(repo, "config", "user.email", "test@example.com")
        self._git(repo, "config", "user.name", "Test User")

        (repo / ".gitignore").write_text("ignored.txt\nexpts/\n")
        (repo / "base.py").write_text("base = 1\n")
        self._git(repo, "add", ".gitignore", "base.py")
        self._git(repo, "commit", "-m", "baseline")

        # Same file has a staged version and then a later unstaged version.
        (repo / "base.py").write_text("base = 2\n")
        self._git(repo, "add", "base.py")
        (repo / "base.py").write_text("base = 3\n")

        nested = repo / "helpers"
        nested.mkdir()
        (nested / "helper.py").write_text("HELPER = True\n")
        (repo / "ignored.txt").write_text("do not snapshot\n")

        expt_dir = repo / "expts" / "expt_1"
        expt_dir.mkdir(parents=True)

        notes = snapshot_source_state(expt_dir, repo_dir=repo)
        snapshot = expt_dir / "source_snapshot"

        assert (snapshot / "staged" / "files" / "base.py").read_text() == "base = 2\n"
        assert (snapshot / "unstaged" / "files" / "base.py").read_text() == "base = 3\n"
        assert (snapshot / "untracked" / "files" / "helpers" / "helper.py").read_text() == "HELPER = True\n"
        assert not (snapshot / "untracked" / "files" / "ignored.txt").exists()

        staged_patch = (snapshot / "staged" / "diff.patch").read_text()
        unstaged_patch = (snapshot / "unstaged" / "diff.patch").read_text()
        assert "+base = 2" in staged_patch
        assert "-base = 2" in unstaged_patch
        assert "+base = 3" in unstaged_patch

        assert "Staged tracked changes: 1 files" in notes
        assert "Unstaged tracked changes: 1 files" in notes
        assert "Untracked, non-ignored files: 1 files" in notes
        assert "    base.py" in notes
        assert "    helpers/helper.py" in notes
        assert "ignored.txt" not in notes
