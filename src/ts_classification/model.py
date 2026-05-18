from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class ClassificationResult:
    clf: RandomForestClassifier
    accuracy: float
    y_test: pd.Series
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    report: str


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict[str, Any],
) -> ClassificationResult:
    model_cfg = cfg.get("model") or {}
    clf = RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 100)),
        max_depth=int(model_cfg.get("max_depth", 10)),
        random_state=int(model_cfg.get("random_state", 42)),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Down", "Up"])
    return ClassificationResult(
        clf=clf,
        accuracy=accuracy,
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        report=report,
    )
