from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from ts_classification import __version__
from ts_classification.config import configure_logging, load_config
from ts_classification.data import (
    FEATURE_COLUMNS,
    build_feature_frame,
    make_synthetic_series,
    temporal_train_test_split,
)
from ts_classification.model import train_classifier
from ts_classification.paths import (
    DEFAULT_CONFIG_PATH,
    path_relative_to_project,
    resolve_project_path,
)
from ts_classification.plots import save_classification_plots

logger = logging.getLogger(__name__)


def run(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    cfg = load_config(path)
    configure_logging(cfg)
    series = make_synthetic_series(cfg)
    df = build_feature_frame(series)
    data_cfg = cfg.get("data") or {}
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    train_df, test_df, split_idx = temporal_train_test_split(df, train_ratio)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]
    result = train_classifier(X_train, y_train, X_test, y_test, cfg)
    logger.info("Accuracy: %.3f", result.accuracy)
    logger.info("Train samples: %d | Test samples: %d", len(X_train), len(X_test))
    logger.info("Class distribution - Train: %s", y_train.value_counts().to_dict())
    logger.info("Class distribution - Test:  %s", y_test.value_counts().to_dict())
    logger.info("Classification Report:\n%s", result.report)
    figures = save_classification_plots(df, split_idx, test_df, result, cfg)
    out_cfg = cfg.get("output") or {}
    results_path = resolve_project_path(out_cfg.get("results_path", "outputs/results.json"))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": __version__,
        "accuracy": result.accuracy,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "figures": {k: path_relative_to_project(v) for k, v in figures.items()},
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", results_path)
    return {"df": df, "result": result, "figures": figures, "results_path": results_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time series direction classification (Random Forest demo)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
