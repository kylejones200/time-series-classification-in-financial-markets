import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    n_splits: int = 5
    season: int = 12
    max_lag: int = 12


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def build_supervised(s: pd.Series, max_lag: int, season: int) -> pd.DataFrame:
    df = pd.DataFrame({'y': s})
    # Lags
    for k in range(1, max_lag + 1):
        df[f'lag{k}'] = df['y'].shift(k)
    # Seasonal naive
    df['season_lag'] = df['y'].shift(season)
    # Calendar features
    m = df.index.month
    df['sin12'] = np.sin(2 * np.pi * m / 12.0)
    df['cos12'] = np.cos(2 * np.pi * m / 12.0)
    # Next-month direction label (binary)
    df['y_next'] = df['y'].shift(-1)
    df['up'] = (df['y_next'] > df['y']).astype(int)
    df = df.dropna()
    return df


def chrono_classification(df: pd.DataFrame, cfg: Config):
    features = [c for c in df.columns if c not in ('y','y_next','up')]
    X = df[features].values
    y = df['up'].values
    idx = np.arange(len(df))
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    accs, aucs = [], []
    for tr, te in tscv.split(idx):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:,1]
        pred = (proba >= 0.5).astype(int)
        accs.append(accuracy_score(y_te, pred))
        try:
            aucs.append(roc_auc_score(y_te, proba))
        except Exception:
            pass
    return float(np.mean(accs)), (float(np.mean(aucs)) if aucs else np.nan)


def main():
    cfg = Config()
    s = load_series(cfg)
    df = build_supervised(s, cfg.max_lag, cfg.season)
    acc, auc = chrono_classification(df, cfg)
    print(f"Up/Down classification — Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # Simple visualization of last 3 years with predicted direction baseline (seasonal naive)
    tail = s.tail(36)
    plt.figure(figsize=(9,4))
    plt.plot(tail.index, tail.values, label='EIA')
    plt.legend()
    save_fig('eia_cls_updown.png')

if __name__ == '__main__':
    main()
