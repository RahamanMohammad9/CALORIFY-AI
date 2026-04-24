from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_FIGURES_DIR = PROJECT_ROOT / "report_figures"

TEST_SPLIT_PATH = PROJECT_ROOT / "test_split.csv"
LABEL_PATH = MODELS_DIR / "label_classes.txt"
TRAIN_HISTORY_CSV = MODELS_DIR / "training_history.csv"
TRAIN_HISTORY_JSON = MODELS_DIR / "training_history.json"

MODEL_PATH = MODELS_DIR / "resnet50_best.pth"
MEAL_DB_PATH = PROJECT_ROOT / "meal_history.db"

RESULTS_DIR.mkdir(exist_ok=True)
REPORT_FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# Settings
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============================================================
# Dataset
# ============================================================
class FoodDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        img_path = str(row["image"])
        label = int(row["label_id"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# Helpers
# ============================================================
def ensure_required_files() -> None:
    required = [TEST_SPLIT_PATH, LABEL_PATH, MODEL_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n" + "\n".join(missing)
        )


def load_test_dataframe() -> pd.DataFrame:
    df = pd.read_csv(TEST_SPLIT_PATH)
    needed = {"image", "label_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"test_split.csv missing columns: {sorted(missing)}")

    df = df[df["image"].apply(lambda x: Path(str(x)).exists())].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows found in test_split.csv after filtering.")
    return df


def load_class_names() -> list[str]:
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError("label_classes.txt is empty.")
    return classes


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def load_model(device: torch.device, num_classes: int) -> nn.Module:
    model = build_model(num_classes)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def evaluate_model() -> dict[str, Any]:
    ensure_required_files()

    class_names = load_class_names()
    num_classes = len(class_names)

    test_df = load_test_dataframe()
    test_dataset = FoodDataset(test_df, transform=EVAL_TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, num_classes)

    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    accuracy = accuracy_score(y_true, y_pred)
    top3 = top_k_accuracy_score(
        y_true,
        y_prob,
        k=min(3, num_classes),
        labels=list(range(num_classes)),
    )
    top5 = top_k_accuracy_score(
        y_true,
        y_prob,
        k=min(5, num_classes),
        labels=list(range(num_classes)),
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc_micro = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
    pr_auc_micro = auc(recall, precision)

    max_conf = y_prob.max(axis=1)
    correct = (y_true == y_pred).astype(float)

    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    bin_acc: list[float] = []
    bin_conf: list[float] = []

    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (max_conf > b0) & (max_conf <= b1)
        if np.any(mask):
            acc_bin = float(np.mean(correct[mask]))
            conf_bin = float(np.mean(max_conf[mask]))
            weight = float(np.mean(mask))
            ece += abs(acc_bin - conf_bin) * weight
            bin_acc.append(acc_bin)
            bin_conf.append(conf_bin)

    brier = float(np.mean([
        brier_score_loss(y_bin[:, i], y_prob[:, i]) for i in range(y_bin.shape[1])
    ]))

    return {
        "class_names": class_names,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy,
        "top3": top3,
        "top5": top5,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc_micro": roc_auc_micro,
        "precision_curve": precision,
        "recall_curve": recall,
        "pr_auc_micro": pr_auc_micro,
        "ece": ece,
        "brier": brier,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
    }


# ============================================================
# Plotting functions
# ============================================================
def save_training_history_plot() -> None:
    history_df = None

    if TRAIN_HISTORY_CSV.exists():
        history_df = pd.read_csv(TRAIN_HISTORY_CSV)
    elif TRAIN_HISTORY_JSON.exists():
        with open(TRAIN_HISTORY_JSON, "r", encoding="utf-8") as f:
            history = json.load(f)
        history_df = pd.DataFrame(history)

    if history_df is None or history_df.empty:
        print("training_history.csv/json not found. Skipping training history plot.")
        return

    epochs = np.arange(1, len(history_df) + 1)

    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, history_df["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history_df["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "figure_training_validation_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, history_df["train_loss"], label="Train Loss")
    plt.plot(epochs, history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "figure_training_validation_loss.png", dpi=300)
    plt.close()


def save_performance_bar_chart(metrics: dict[str, Any]) -> None:
    labels = ["Accuracy", "Top-3 Accuracy", "Top-5 Accuracy", "Macro F1", "Weighted F1", "ROC-AUC"]
    values = [
        metrics["accuracy"] * 100,
        metrics["top3"] * 100,
        metrics["top5"] * 100,
        metrics["macro_f1"] * 100,
        metrics["weighted_f1"] * 100,
        metrics["roc_auc_micro"] * 100,
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)
    plt.ylabel("Score (%)")
    plt.title("Overall Model Performance")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "figure_model_performance_summary.png", dpi=300)
    plt.close()


def save_confusion_matrix(metrics: dict[str, Any]) -> None:
    cm = metrics["cm"]

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "figure_confusion_matrix.png", dpi=300)
    plt.close()


def save_roc_curve(metrics: dict[str, Any]) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot(
        metrics["fpr"],
        metrics["tpr"],
        label=f"Micro-average ROC (AUC = {metrics['roc_auc_micro']:.4f})",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "figure_roc_curve.png", dpi=300)
    plt.close()


def save_pr_curve(metrics: dict[str, Any]) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot(
        metrics["recall_curve"],
        metrics["precision_curve"],
        label=f"PR AUC = {metrics['pr_auc_micro']:.4f}",
        linewidth=2,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "figure_precision_recall_curve.png", dpi=300)
    plt.close()


def save_calibration_curve(metrics: dict[str, Any]) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", label="Perfect Calibration")
    plt.plot(
        metrics["bin_conf"],
        metrics["bin_acc"],
        marker="o",
        label=f"Model (ECE = {metrics['ece']:.4f})",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "figure_calibration_curve.png", dpi=300)
    plt.close()


def save_topk_chart(metrics: dict[str, Any]) -> None:
    labels = ["Top-1", "Top-3", "Top-5"]
    values = [metrics["accuracy"] * 100, metrics["top3"] * 100, metrics["top5"] * 100]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, values)
    plt.ylabel("Accuracy (%)")
    plt.title("Top-K Accuracy")
    plt.ylim(0, 100)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "figure_topk_accuracy.png", dpi=300)
    plt.close()


def save_dashboard_charts_from_db() -> None:
    if not MEAL_DB_PATH.exists():
        print("meal_history.db not found. Skipping dashboard charts.")
        return

    conn = sqlite3.connect(MEAL_DB_PATH)

    calorie_df = pd.read_sql_query(
        """
        SELECT substr(created_at, 1, 10) AS day, SUM(calories) AS total_calories
        FROM meals
        GROUP BY substr(created_at, 1, 10)
        ORDER BY day ASC
        """,
        conn,
    )

    macro_df = pd.read_sql_query(
        """
        SELECT
            SUM(protein) AS protein,
            SUM(carbs) AS carbs,
            SUM(fat) AS fat
        FROM meals
        WHERE substr(created_at, 1, 10) = date('now', 'localtime')
        """,
        conn,
    )

    conn.close()

    if not calorie_df.empty:
        plt.figure(figsize=(10, 5.5))
        plt.plot(calorie_df["day"], calorie_df["total_calories"], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Date")
        plt.ylabel("Calories")
        plt.title("Daily Calorie Trend")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORT_FIGURES_DIR / "figure_daily_calorie_trend.png", dpi=300)
        plt.close()

    if not macro_df.empty:
        protein = float(macro_df.loc[0, "protein"] or 0)
        carbs = float(macro_df.loc[0, "carbs"] or 0)
        fat = float(macro_df.loc[0, "fat"] or 0)

        if (protein + carbs + fat) > 0:
            plt.figure(figsize=(6.5, 6.5))
            plt.pie(
                [protein, carbs, fat],
                labels=["Protein", "Carbs", "Fat"],
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title("Daily Macronutrient Distribution")
            plt.tight_layout()
            plt.savefig(REPORT_FIGURES_DIR / "figure_macro_distribution.png", dpi=300)
            plt.close()


def save_system_architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    boxes = [
        (0.6, 7.8, 2.0, 1.0, "User Input\n(Image / Text / Voice-style)"),
        (3.1, 7.8, 2.0, 1.0, "Food Recognition\n(ResNet50 Model)"),
        (5.6, 7.8, 2.0, 1.0, "Nutrition Engine\n(Local CSV + API)"),
        (8.1, 7.8, 1.3, 1.0, "Output\nPrediction"),
        (3.1, 5.4, 2.0, 1.0, "Personalisation\n(BMI / BMR / Goals)"),
        (5.6, 5.4, 2.0, 1.0, "Behaviour Tracking\nSleep / Water / Activity / Weight"),
        (3.1, 3.0, 2.0, 1.0, "SQLite Database"),
        (5.6, 3.0, 2.0, 1.0, "AI Insights Engine"),
        (8.1, 3.0, 1.3, 1.0, "Dashboard\n(Streamlit)"),
    ]

    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((2.6, 8.3), (3.1, 8.3)),
        ((5.1, 8.3), (5.6, 8.3)),
        ((7.6, 8.3), (8.1, 8.3)),
        ((4.1, 7.8), (4.1, 6.4)),
        ((6.6, 7.8), (6.6, 6.4)),
        ((4.1, 5.4), (4.1, 4.0)),
        ((6.6, 5.4), (6.6, 4.0)),
        ((5.1, 3.5), (5.6, 3.5)),
        ((7.6, 3.5), (8.1, 3.5)),
    ]

    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.6))

    plt.title("Calorify AI System Architecture", fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORT_FIGURES_DIR / "figure_system_architecture.png", dpi=300)
    plt.close()


def save_metrics_table(metrics: dict[str, Any]) -> None:
    table_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Top-3 Accuracy",
                "Top-5 Accuracy",
                "Macro F1",
                "Weighted F1",
                "ROC-AUC",
                "PR-AUC",
                "ECE",
                "Brier Score",
            ],
            "Value": [
                round(metrics["accuracy"], 4),
                round(metrics["top3"], 4),
                round(metrics["top5"], 4),
                round(metrics["macro_f1"], 4),
                round(metrics["weighted_f1"], 4),
                round(metrics["roc_auc_micro"], 4),
                round(metrics["pr_auc_micro"], 4),
                round(metrics["ece"], 4),
                round(metrics["brier"], 4),
            ],
        }
    )
    table_df.to_csv(REPORT_FIGURES_DIR / "report_metrics_summary.csv", index=False)


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("Generating report figures...")

    save_training_history_plot()

    metrics = evaluate_model()
    save_metrics_table(metrics)

    save_performance_bar_chart(metrics)
    save_confusion_matrix(metrics)
    save_roc_curve(metrics)
    save_pr_curve(metrics)
    save_calibration_curve(metrics)
    save_topk_chart(metrics)

    save_dashboard_charts_from_db()
    save_system_architecture_diagram()

    print(f"Done. All files saved in: {REPORT_FIGURES_DIR}")


if __name__ == "__main__":
    main()