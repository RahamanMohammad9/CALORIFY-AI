import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
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
    if not os.path.exists(TEST_SPLIT_PATH):
        raise FileNotFoundError(f"Missing file: {TEST_SPLIT_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing file: {MODEL_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Missing file: {LABELS_PATH}")

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

    y_true = []
    y_prob = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # ROC curve (micro-average one-vs-rest)
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"Micro-average ROC (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiclass, One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    # "MUI" curve: micro-average Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"Micro-average PR (AUC = {pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("MUI Curve (Micro-average Precision-Recall)")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mui_curve.png"), dpi=300)
    plt.close()

    with open(os.path.join(RESULTS_DIR, "roc_mui_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"ROC AUC (micro): {roc_auc:.6f}\n")
        f.write(f"MUI/PR AUC (micro): {pr_auc:.6f}\n")

    print("Saved:")
    print(os.path.join(RESULTS_DIR, "roc_curve.png"))
    print(os.path.join(RESULTS_DIR, "mui_curve.png"))
    print(os.path.join(RESULTS_DIR, "roc_mui_summary.txt"))


if __name__ == "__main__":
    main()
