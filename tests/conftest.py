from pathlib import Path
import importlib.util
import sys
import types

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


_REAL_TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

if not _REAL_TF_AVAILABLE:
    tf_stub = types.ModuleType("tensorflow")

    class _KerasUtils:
        @staticmethod
        def set_random_seed(seed):
            return None

    class _Experimental:
        @staticmethod
        def enable_op_determinism():
            return None

    class _Config:
        experimental = _Experimental()

    class _Keras:
        utils = _KerasUtils()

    tf_stub.keras = _Keras()
    tf_stub.config = _Config()
    sys.modules.setdefault("tensorflow", tf_stub)


@pytest.fixture
def tf():
    if not _REAL_TF_AVAILABLE:
        pytest.skip("tensorflow not installed")
    import tensorflow as tf  # type: ignore
    return tf
