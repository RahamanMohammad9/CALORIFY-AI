import os
import json
import random

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "food_model.pth")
LABELS_PATH = os.path.join(BASE_DIR, "models", "label_classes.txt")

TRAIN_SPLIT_CSV = os.path.join(BASE_DIR, "train_split.csv")
TEST_SPLIT_CSV = os.path.join(BASE_DIR, "test_split.csv")

# Optional history files
HISTORY_JSON = os.path.join(BASE_DIR, "models", "training_history.json")
HISTORY_CSV = os.path.join(BASE_DIR, "models", "training_history.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "report_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0
TOP_K_SAMPLES = 12
SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# DATASET
# =========================================================
class FoodCSVDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.dataframe = pd.read_csv(csv_path)
        self.csv_dir = os.path.dirname(os.path.abspath(csv_path))

        required_cols = {"image", "label_id"}
        if not required_cols.issubset(self.dataframe.columns):
            raise ValueError(f"{csv_path} must contain columns: {required_cols}")

        self.dataframe["image"] = self.dataframe["image"].astype(str).apply(self._resolve_image_path)
        self.dataframe = self.dataframe[self.dataframe["image"].apply(os.path.exists)].reset_index(drop=True)

        if len(self.dataframe) == 0:
            raise ValueError(f"No valid image paths found in {csv_path}")

        self.transform = transform

    def _resolve_image_path(self, img_path: str) -> str:
        # Support absolute paths and CSV-relative paths.
        normalized = os.path.normpath(img_path)
        if os.path.isabs(normalized):
            return normalized
        return os.path.normpath(os.path.join(self.csv_dir, normalized))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, "image"]
        label = int(self.dataframe.loc[idx, "label_id"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# HELPERS
# =========================================================
def load_labels(labels_path: str):
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    if not labels:
        raise ValueError("Label file is empty.")

    return labels


def build_model(num_classes: int):
    model = models.resnet50(weights=None)

    # Match training setup
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def load_model(model_path: str, num_classes: int, device: torch.device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = build_model(num_classes)
    state = torch.load(model_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_csv_dataloader(csv_path: str):
    dataset = FoodCSVDataset(csv_path, transform=get_transform())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    return dataset, loader


def save_plot(fig, filename: str):
    full_path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {full_path}")


def load_history():
    if os.path.exists(HISTORY_JSON):
        with open(HISTORY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    if os.path.exists(HISTORY_CSV):
        df = pd.read_csv(HISTORY_CSV)
        return {
            "train_loss": df["train_loss"].tolist() if "train_loss" in df.columns else [],
            "val_loss": df["val_loss"].tolist() if "val_loss" in df.columns else [],
            "train_acc": df["train_acc"].tolist() if "train_acc" in df.columns else [],
            "val_acc": df["val_acc"].tolist() if "val_acc" in df.columns else [],
        }

    return None


# =========================================================
# PLOTTING FUNCTIONS
# =========================================================
def plot_training_curves(history):
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if train_acc and val_acc:
        epochs = range(1, len(train_acc) + 1)
        fig = plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_plot(fig, "training_validation_accuracy.png")

    if train_loss and val_loss:
        epochs = range(1, len(train_loss) + 1)
        fig = plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_plot(fig, "training_validation_loss.png")


def plot_class_distribution_from_csv(csv_path: str, class_names, title: str, filename: str):
    if not os.path.exists(csv_path):
        print(f"Skipped class distribution plot. CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "label_id" not in df.columns:
        print(f"Skipped class distribution plot. 'label_id' missing in {csv_path}")
        return

    counts = df["label_id"].value_counts().sort_index()

    labels = []
    values = []
    for i in range(len(class_names)):
        labels.append(class_names[i])
        values.append(int(counts.get(i, 0)))

    fig = plt.figure(figsize=(14, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(fig, filename)


def evaluate_model(model, loader, class_names, device):
    y_true = []
    y_pred = []
    y_probs = []
    sample_records = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        total_batches = len(loader)
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = softmax(outputs)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy().tolist())

            for i in range(images.size(0)):
                sample_records.append({
                    "image_tensor": images[i].cpu(),
                    "true_idx": labels[i].item(),
                    "pred_idx": preds[i].item(),
                    "confidence": probs[i][preds[i]].item()
                })

            if batch_idx % 25 == 0 or batch_idx == total_batches:
                print(f"Processed batch {batch_idx}/{total_batches}")

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    report_df.to_csv(report_csv_path, index=True)
    print(f"Saved: {report_csv_path}")

    return np.array(y_true), np.array(y_pred), np.array(y_probs), sample_records, accuracy


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    save_plot(fig, "confusion_matrix.png")

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig = plt.figure(figsize=(12, 10))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    save_plot(fig, "normalized_confusion_matrix.png")


def plot_per_class_accuracy(y_true, y_pred, class_names):
    accuracies = []
    for i, _ in enumerate(class_names):
        mask = (y_true == i)
        if np.sum(mask) == 0:
            accuracies.append(0.0)
        else:
            acc = np.mean(y_pred[mask] == y_true[mask])
            accuracies.append(acc * 100)

    sort_idx = np.argsort(accuracies)[::-1]
    sorted_labels = [class_names[i] for i in sort_idx]
    sorted_acc = [accuracies[i] for i in sort_idx]

    fig = plt.figure(figsize=(14, 6))
    plt.bar(sorted_labels, sorted_acc)
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(fig, "per_class_accuracy.png")


def plot_confidence_histogram(y_probs, y_pred):
    confidences = [y_probs[i][y_pred[i]] for i in range(len(y_pred))]
    fig = plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=20)
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Samples")
    plt.title("Prediction Confidence Distribution")
    plt.grid(True, alpha=0.3)
    save_plot(fig, "prediction_confidence_histogram.png")


def unnormalize_image(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def plot_sample_predictions(sample_records, class_names, num_samples=12):
    if not sample_records:
        return

    incorrect = [s for s in sample_records if s["true_idx"] != s["pred_idx"]]
    correct = [s for s in sample_records if s["true_idx"] == s["pred_idx"]]

    selected = []
    selected.extend(random.sample(incorrect, min(len(incorrect), num_samples // 2)))
    remaining = num_samples - len(selected)
    selected.extend(random.sample(correct, min(len(correct), remaining)))

    if not selected:
        return

    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    fig = plt.figure(figsize=(12, 4 * rows))

    for i, record in enumerate(selected, start=1):
        ax = fig.add_subplot(rows, cols, i)
        img = unnormalize_image(record["image_tensor"])
        ax.imshow(img)
        true_label = class_names[record["true_idx"]]
        pred_label = class_names[record["pred_idx"]]
        conf = record["confidence"] * 100
        ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {conf:.1f}%")
        ax.axis("off")

    save_plot(fig, "sample_predictions.png")


def save_summary_txt(accuracy, class_names, y_true):
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Calorify AI Report Graph Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Overall test accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Number of classes: {len(class_names)}\n")
        f.write(f"Number of test samples: {len(y_true)}\n\n")
        f.write("Generated files:\n")
        f.write("- training_validation_accuracy.png (if history exists)\n")
        f.write("- training_validation_loss.png (if history exists)\n")
        f.write("- confusion_matrix.png\n")
        f.write("- normalized_confusion_matrix.png\n")
        f.write("- per_class_accuracy.png\n")
        f.write("- prediction_confidence_histogram.png\n")
        f.write("- sample_predictions.png\n")
        f.write("- classification_report.csv\n")
        f.write("- class_distribution_train.png\n")
    print(f"Saved: {summary_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("Generating report graphs for Calorify AI...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"TRAIN_SPLIT_CSV: {TRAIN_SPLIT_CSV}")
    print(f"TEST_SPLIT_CSV: {TEST_SPLIT_CSV}")

    class_names = load_labels(LABELS_PATH)
    print(f"Loaded {len(class_names)} classes.")

    history = load_history()
    if history is not None:
        print("Training history found. Generating training curves...")
        plot_training_curves(history)
    else:
        print("No training_history.json or training_history.csv found. Skipping training curves.")

    plot_class_distribution_from_csv(
        TRAIN_SPLIT_CSV,
        class_names,
        title="Training Dataset Class Distribution",
        filename="class_distribution_train.png"
    )

    test_dataset, test_loader = get_csv_dataloader(TEST_SPLIT_CSV)
    print(f"Loaded test samples: {len(test_dataset)}")

    model = load_model(MODEL_PATH, len(class_names), device)

    y_true, y_pred, y_probs, sample_records, accuracy = evaluate_model(
        model, test_loader, class_names, device
    )

    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_per_class_accuracy(y_true, y_pred, class_names)
    plot_confidence_histogram(y_probs, y_pred)
    plot_sample_predictions(sample_records, class_names, num_samples=TOP_K_SAMPLES)
    save_summary_txt(accuracy, class_names, y_true)

    print("\nDone. All graphs saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()