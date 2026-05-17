from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ts_classification.data import FEATURE_COLUMNS
from ts_classification.model import ClassificationResult
from ts_classification.paths import FIGURES_DIR


def _figure_paths(cfg: dict[str, Any]) -> tuple[tuple[Path, Path], int]:
    out_cfg = cfg.get("output") or {}
    figures_dir = FIGURES_DIR
    if rel := out_cfg.get("figures_dir"):
        from ts_classification.paths import resolve_project_path

        figures_dir = resolve_project_path(rel)
    figures_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(out_cfg.get("figure_dpi", 300))
    return (
        figures_dir / "classification_ts_analysis.png",
        figures_dir / "classification_probabilities.png",
    ), dpi


def save_classification_plots(
    df: pd.DataFrame,
    split_idx: int,
    test_df: pd.DataFrame,
    result: ClassificationResult,
    cfg: dict[str, Any],
) -> dict[str, Path]:
    """Write analysis dashboard and probability timeline to outputs/figures."""
    out_cfg = cfg.get("output") or {}
    figsize = tuple(out_cfg.get("figsize", [14, 10]))
    show = bool(out_cfg.get("show", False))
    (analysis_path, proba_path), dpi = _figure_paths(cfg)

    features = FEATURE_COLUMNS
    y_test = result.y_test
    y_pred = result.y_pred
    y_pred_proba = result.y_pred_proba
    acc = result.accuracy
    clf = result.clf

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].plot(
        df.index[:split_idx],
        df["value"][:split_idx],
        color="#2c3e50",
        linewidth=1,
        label="Train",
    )
    axes[0, 0].plot(
        df.index[split_idx:],
        df["value"][split_idx:],
        color="#e74c3c",
        linewidth=1,
        label="Test",
    )
    axes[0, 0].axvline(split_idx, color="red", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Time Series with Train/Test Split", fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].set_ylabel("Value")

    test_indices = test_df.index
    correct = y_test == y_pred
    incorrect = ~correct
    axes[0, 1].scatter(
        test_indices[correct],
        test_df["value"][correct],
        c="#27ae60",
        s=30,
        alpha=0.6,
        label="Correct",
    )
    axes[0, 1].scatter(
        test_indices[incorrect],
        test_df["value"][incorrect],
        c="#e74c3c",
        s=30,
        alpha=0.6,
        label="Incorrect",
    )
    axes[0, 1].set_title(f"Classification Performance (Accuracy: {acc:.3f})", fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].set_xlabel("Index")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1, 0],
        xticklabels=["Down", "Up"],
        yticklabels=["Down", "Up"],
        cbar_kws={"label": "Count"},
    )
    axes[1, 0].set_title("Confusion Matrix", fontsize=12)
    axes[1, 0].set_ylabel("Actual")
    axes[1, 0].set_xlabel("Predicted")

    importances = clf.feature_importances_
    feature_imp_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=True
    )
    axes[1, 1].barh(
        feature_imp_df["feature"],
        feature_imp_df["importance"],
        color="#3498db",
        alpha=0.7,
    )
    axes[1, 1].set_title("Feature Importance", fontsize=12)
    axes[1, 1].set_xlabel("Importance")

    plt.tight_layout()
    fig.savefig(analysis_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(test_indices, y_pred_proba, color="#3498db", linewidth=1.5, alpha=0.7)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.fill_between(
        test_indices,
        0.5,
        y_pred_proba,
        where=(y_pred_proba >= 0.5),
        alpha=0.3,
        color="#27ae60",
        label="Predicted Up",
    )
    ax.fill_between(
        test_indices,
        y_pred_proba,
        0.5,
        where=(y_pred_proba < 0.5),
        alpha=0.3,
        color="#e74c3c",
        label="Predicted Down",
    )
    ax.set_title("Prediction Probabilities Over Time", fontsize=12)
    ax.set_ylabel("P(Up)")
    ax.set_xlabel("Index")
    ax.legend()
    plt.tight_layout()
    fig.savefig(proba_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"analysis": analysis_path, "probabilities": proba_path}
