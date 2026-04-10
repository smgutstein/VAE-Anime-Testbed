import pytest

from test_helpers import make_config_ini_text, write_config_text, write_minimal_config


class TestConfigLoading:
    def test_config_loads_with_new_kl_names(self, tmp_path):
        from VAE_Anime_Config import TrainerConfig

        config_path = tmp_path / "config_new.ini"
        write_minimal_config(config_path, parent_dir=tmp_path / "expts", use_legacy_kl_names=False)
        cfg = TrainerConfig.from_file(config_path)
        assert cfg.initial_kl_weight == 1e-6
        assert cfg.max_kl_weight == 10.0
        assert cfg.kl_weight_update_factor == 1.2

    def test_config_loads_with_legacy_kl_names(self, tmp_path):
        from VAE_Anime_Config import TrainerConfig

        config_path = tmp_path / "config_old.ini"
        write_minimal_config(config_path, parent_dir=tmp_path / "expts", use_legacy_kl_names=True)
        cfg = TrainerConfig.from_file(config_path)
        assert cfg.initial_kl_weight == 1e-6
        assert cfg.max_kl_weight == 10.0
        assert cfg.kl_weight_update_factor == 1.2


class TestLoadAndValidateConfig:
    @pytest.fixture(autouse=True)
    def _import(self):
        from utils import load_and_validate_config
        self.load = load_and_validate_config

    def test_valid_fixed_beta_config(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(loss_policy="fixed_beta", beta="0.5"))
        config = self.load(cfg_file)
        assert config.get("Training_Parameters", "loss_policy") == "fixed_beta"

    def test_valid_adaptive_kl_config_new_keys(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(
            loss_policy="adaptive_kl",
            initial_kl_weight="0.01",
            max_kl_weight="1.0",
            kl_weight_update_factor="0.1",
            running_window="10",
        ))
        assert self.load(cfg_file).get("Training_Parameters", "loss_policy") == "adaptive_kl"

    def test_valid_adaptive_kl_config_legacy_keys(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(
            loss_policy="adaptive_kl",
            kl_adj_factor="0.01",
            kl_adj_factor_max="1.0",
            kl_adj_update_factor="0.1",
            running_window="10",
        ))
        assert self.load(cfg_file).get("Training_Parameters", "loss_policy") == "adaptive_kl"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self.load(tmp_path / "missing.ini")

    @pytest.mark.parametrize(
        "section,flag",
        [
            ("Training_Parameters", "include_training"),
            ("Output_Parameters", "include_output"),
            ("Data_Parameters", "include_data"),
            ("Model_Parameters", "include_model"),
            ("Monitoring_Parameters", "include_monitoring"),
        ],
    )
    def test_missing_required_section_raises(self, tmp_path, section, flag):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(**{flag: False}))
        with pytest.raises(ValueError, match=section):
            self.load(cfg_file)

    def test_unknown_loss_policy_raises(self, tmp_path):
        text = make_config_ini_text().replace("loss_policy = fixed_beta", "loss_policy = magic").replace("beta = 0.5", "")
        with pytest.raises(ValueError, match="loss_policy"):
            self.load(write_config_text(tmp_path, text))

    def test_fixed_beta_missing_beta_raises(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(loss_policy="fixed_beta", beta=None))
        with pytest.raises(ValueError, match="beta"):
            self.load(cfg_file)

    def test_adaptive_kl_missing_running_window_raises(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(
            loss_policy="adaptive_kl",
            initial_kl_weight="0.01",
            max_kl_weight="1.0",
            kl_weight_update_factor="0.1",
            running_window=None,
        ))
        with pytest.raises(ValueError, match="running_window"):
            self.load(cfg_file)

    def test_adaptive_kl_missing_all_kl_keys_raises(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(loss_policy="adaptive_kl", running_window="10"))
        with pytest.raises(ValueError, match="Missing KL-weight config options"):
            self.load(cfg_file)


class TestTrainerConfig:
    @pytest.fixture(autouse=True)
    def _import(self):
        from VAE_Anime_Config import TrainerConfig
        self.TrainerConfig = TrainerConfig

    def test_fixed_beta_fields_have_correct_types(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(loss_policy="fixed_beta", beta="0.7", epochs="5"))
        cfg = self.TrainerConfig.from_file(cfg_file)
        assert isinstance(cfg.epochs, int)
        assert isinstance(cfg.learning_rate, float)
        assert isinstance(cfg.beta, float)
        assert cfg.beta == pytest.approx(0.7)
        assert cfg.initial_kl_weight is None
        assert cfg.max_kl_weight is None
        assert cfg.running_window is None

    def test_adaptive_kl_fields_have_correct_types(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(
            loss_policy="adaptive_kl",
            initial_kl_weight="0.01",
            max_kl_weight="1.0",
            kl_weight_update_factor="0.05",
            running_window="20",
        ))
        cfg = self.TrainerConfig.from_file(cfg_file)
        assert isinstance(cfg.initial_kl_weight, float)
        assert isinstance(cfg.kl_weight_update_factor, float)
        assert isinstance(cfg.running_window, int)
        assert cfg.running_window == 20
        assert cfg.beta is None

    def test_safety_fields_load_with_correct_types(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text())
        cfg = self.TrainerConfig.from_file(cfg_file)
        assert isinstance(cfg.max_grad_norm, float)
        assert isinstance(cfg.step_guard_kl_jump_ratio_threshold, float)
        assert isinstance(cfg.step_guard_kl_abs_threshold, float)
        assert isinstance(cfg.step_guard_max_log_var_threshold, float)

    def test_filter_factors_parsed_as_tuple_of_ints(self, tmp_path):
        cfg = self.TrainerConfig.from_file(write_config_text(tmp_path, make_config_ini_text(filter_factors="1, 2, 4")))
        assert cfg.filter_factors == (1, 2, 4)

    def test_legacy_kl_keys_map_correctly(self, tmp_path):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(
            loss_policy="adaptive_kl",
            kl_adj_factor="0.02",
            kl_adj_factor_max="0.8",
            kl_adj_update_factor="0.05",
            running_window="15",
        ))
        cfg = self.TrainerConfig.from_file(cfg_file)
        assert cfg.initial_kl_weight == pytest.approx(0.02)
        assert cfg.max_kl_weight == pytest.approx(0.8)
        assert cfg.kl_weight_update_factor == pytest.approx(0.05)

    def test_config_is_frozen(self, tmp_path):
        cfg = self.TrainerConfig.from_file(write_config_text(tmp_path, make_config_ini_text()))
        with pytest.raises((AttributeError, TypeError)):
            cfg.epochs = 999

    @pytest.mark.parametrize(
        "override,match",
        [
            ({"epochs": "0"}, "epochs"),
            ({"learning_rate": "-0.001"}, "learning_rate"),
            ({"val_split": "0.0"}, "val_split"),
            ({"val_split": "1.0"}, "val_split"),
            ({"beta": "-0.1"}, "beta"),
            ({"filter_factors": "1, 2"}, "filter_factors"),
            ({"filter_factors": "1, 2, 4, 8"}, "filter_factors"),
        ],
    )
    def test_invalid_field_values_raise(self, tmp_path, override, match):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(**override))
        with pytest.raises(ValueError, match=match):
            self.TrainerConfig.from_file(cfg_file)

    @pytest.mark.parametrize(
        "override,match",
        [
            ({"max_grad_norm": "0"}, "max_grad_norm"),
            ({"step_guard_kl_jump_ratio_threshold": "0"}, "step_guard_kl_jump_ratio_threshold"),
            ({"step_guard_kl_abs_threshold": "0"}, "step_guard_kl_abs_threshold"),
            ({"step_guard_max_log_var_threshold": "0"}, "step_guard_max_log_var_threshold"),
        ],
    )
    def test_invalid_safety_fields_raise(self, tmp_path, override, match):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(**override))
        with pytest.raises(ValueError, match=match):
            self.TrainerConfig.from_file(cfg_file)

    @pytest.mark.parametrize(
        "override,match",
        [
            ({
                "initial_kl_weight": "1.0",
                "max_kl_weight": "0.5",
                "kl_weight_update_factor": "0.1",
                "running_window": "10",
                "loss_policy": "adaptive_kl",
            }, "max_kl_weight"),
            ({
                "initial_kl_weight": "0.01",
                "max_kl_weight": "1.0",
                "kl_weight_update_factor": "0.1",
                "running_window": "1",
                "loss_policy": "adaptive_kl",
            }, "running_window"),
        ],
    )
    def test_invalid_adaptive_kl_fields_raise(self, tmp_path, override, match):
        cfg_file = write_config_text(tmp_path, make_config_ini_text(**override))
        with pytest.raises(ValueError, match=match):
            self.TrainerConfig.from_file(cfg_file)
