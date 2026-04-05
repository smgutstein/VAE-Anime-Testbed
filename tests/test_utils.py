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
