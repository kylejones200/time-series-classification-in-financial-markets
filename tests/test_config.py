from ts_classification.config import load_config
from ts_classification.paths import DEFAULT_CONFIG_PATH


def test_load_config_has_expected_sections():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert "data" in cfg
    assert "model" in cfg
    assert "output" in cfg
    assert cfg["data"]["n_samples"] == 500
