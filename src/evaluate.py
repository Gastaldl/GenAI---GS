"""Tabelas de metricas e figuras (comparacao, matriz de confusao, dispersao)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_metrics_table(df: pd.DataFrame, name: str) -> Path:
    path = ROOT / "reports" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def plot_model_comparison(df: pd.DataFrame, metric: str, title: str, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = df.sort_values(metric, ascending=False)
    ax.barh(ordered["model"], ordered[metric], color="#2a9d8f")
    ax.set_xlabel(metric)
    ax.set_title(title)
    for i, v in enumerate(ordered[metric]):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.invert_yaxis()
    fig.tight_layout()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion(y_true, y_pred, labels, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax, xticks_rotation=30
    )
    ax.set_title("Matriz de Confusão - melhor classificador")
    fig.tight_layout()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_regression_scatter(y_true, y_pred, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color="#264653")
    lims = [
        float(np.min([np.min(y_true), np.min(y_pred)])),
        float(np.max([np.max(y_true), np.max(y_pred)])),
    ]
    ax.plot(lims, lims, "--", color="#e76f51", label="ideal")
    ax.set_xlabel("Variação real (% ao ano)")
    ax.set_ylabel("Variação prevista (% ao ano)")
    ax.set_title("Regressão: previsto × real (melhor modelo)")
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
