import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "food_model.pth")
LABELS_PATH = os.path.join(BASE_DIR, "models", "label_classes.txt")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "test_split.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMG_SIZE = 224
BATCH_SIZE = 64

os.makedirs(RESULTS_DIR, exist_ok=True)


class FoodDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, "image"]
        label = int(self.dataframe.loc[idx, "label_id"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = [x.strip() for x in f if x.strip()]
    num_classes = len(class_names)

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    test_df = test_df[test_df["image"].apply(os.path.exists)].reset_index(drop=True)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(FoodDataset(test_df, transform=tfm), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # Save raw matrix CSV (detailed machine-readable)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix_detailed.csv"), encoding="utf-8")

    # Save normalized matrix CSV
    cmn_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    cmn_df.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix_normalized.csv"), encoding="utf-8")

    # Full labeled confusion matrix image
    fig = plt.figure(figsize=(36, 32))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Detailed Confusion Matrix (Counts)")
    plt.colorbar(fraction=0.02, pad=0.01)
    ticks = np.arange(num_classes)
    plt.xticks(ticks, class_names, rotation=90, fontsize=5)
    plt.yticks(ticks, class_names, fontsize=5)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_detailed_labeled.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Full normalized labeled confusion matrix image
    fig = plt.figure(figsize=(36, 32))
    plt.imshow(cm_norm, interpolation="nearest", cmap="magma")
    plt.title("Detailed Normalized Confusion Matrix")
    plt.colorbar(fraction=0.02, pad=0.01)
    ticks = np.arange(num_classes)
    plt.xticks(ticks, class_names, rotation=90, fontsize=5)
    plt.yticks(ticks, class_names, fontsize=5)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_normalized_labeled.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Top misclassifications table
    top_confusions = []
    for i in range(num_classes):
        row = cm[i].copy()
        row[i] = 0
        j = int(np.argmax(row))
        if row[j] > 0:
            top_confusions.append({
                "true_class": class_names[i],
                "most_confused_with": class_names[j],
                "count": int(row[j]),
                "row_total": int(cm[i].sum()),
                "confusion_rate": float(row[j] / max(cm[i].sum(), 1)),
            })
    top_df = pd.DataFrame(top_confusions).sort_values(by=["count", "confusion_rate"], ascending=False)
    top_df.to_csv(os.path.join(RESULTS_DIR, "top_misclassifications.csv"), index=False, encoding="utf-8")

    print("Saved detailed confusion matrix outputs in results/")


if __name__ == "__main__":
    main()
