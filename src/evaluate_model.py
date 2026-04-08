import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -----------------------------
# Settings
# -----------------------------
BATCH_SIZE = 32
IMG_SIZE = 224
MODEL_PATH = "models/food_model.pth"
LABEL_PATH = "models/label_classes.txt"
TEST_SPLIT_PATH = "test_split.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Check required files
# -----------------------------
for path in [TEST_SPLIT_PATH, MODEL_PATH, LABEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

# -----------------------------
# Load test split
# -----------------------------
test_df = pd.read_csv(TEST_SPLIT_PATH)
required_cols = {"image", "label_id"}
missing_cols = required_cols - set(test_df.columns)
if missing_cols:
    raise ValueError(f"test_split.csv missing columns: {sorted(missing_cols)}")

test_df = test_df[test_df["image"].apply(os.path.exists)].reset_index(drop=True)

if test_df.empty:
    raise ValueError("No valid test rows found after existence filtering.")

print("Valid test samples:", len(test_df))


# -----------------------------
# Dataset class
# -----------------------------
class FoodDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, "image"]
        label = int(self.dataframe.loc[idx, "label_id"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------------
# Transform
# -----------------------------
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_dataset = FoodDataset(test_df, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Load labels
# -----------------------------
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

num_classes = len(class_names)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load model
# -----------------------------
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# -----------------------------
# Evaluate
# -----------------------------
all_labels = []
all_preds = []
all_probs = []
all_top3 = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        top3 = torch.topk(probs, k=min(3, num_classes), dim=1).indices

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_top3.extend(top3.cpu().numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_prob = np.array(all_probs)

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
top3_acc = top_k_accuracy_score(y_true, y_prob, k=min(3, num_classes), labels=list(range(num_classes)))

print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Top-3 Accuracy: {top3_acc * 100:.2f}%")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16, 14))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

with open(os.path.join(RESULTS_DIR, "accuracy.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {acc * 100:.2f}%\n")
    f.write(f"Top-3 Accuracy: {top3_acc * 100:.2f}%\n")
    f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
    f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")

# -----------------------------
# ROC / PR / calibration
# -----------------------------
y_bin = label_binarize(y_true, classes=list(range(num_classes)))

fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
roc_auc_micro = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"Micro-average ROC (AUC = {roc_auc_micro:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Multiclass, One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close()

precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
pr_auc_micro = auc(recall, precision)
plt.figure(figsize=(7, 6))
plt.plot(recall, precision, label=f"Micro-average PR (AUC = {pr_auc_micro:.4f})", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pr_curve.png"))
plt.close()

per_class_rows = []
for i, class_name in enumerate(class_names):
    y_true_i = y_bin[:, i]
    y_prob_i = y_prob[:, i]
    if len(np.unique(y_true_i)) < 2:
        continue
    fpr_i, tpr_i, _ = roc_curve(y_true_i, y_prob_i)
    prec_i, rec_i, _ = precision_recall_curve(y_true_i, y_prob_i)
    per_class_rows.append({
        "class": class_name,
        "roc_auc": float(auc(fpr_i, tpr_i)),
        "pr_auc": float(auc(rec_i, prec_i)),
        "support": int(np.sum(y_true_i)),
    })

per_class_auc_df = pd.DataFrame(per_class_rows).sort_values("roc_auc", ascending=False)
per_class_auc_df.to_csv(os.path.join(RESULTS_DIR, "per_class_auc.csv"), index=False)

max_conf = y_prob.max(axis=1)
correct = (y_pred == y_true).astype(float)
bins = np.linspace(0.0, 1.0, 11)

ece = 0.0
bin_acc = []
bin_conf = []

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

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
plt.plot(bin_conf, bin_acc, marker="o", label=f"Model (ECE={ece:.4f})")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title("Calibration Curve")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "calibration_curve.png"))
plt.close()

with open(os.path.join(RESULTS_DIR, "advanced_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"ROC AUC (micro): {roc_auc_micro:.6f}\n")
    f.write(f"PR AUC (micro): {pr_auc_micro:.6f}\n")
    f.write(f"ECE (10 bins): {ece:.6f}\n")
    f.write(f"Brier score (macro over classes): {brier:.6f}\n")

# -----------------------------
# Error analysis
# -----------------------------
misclassified = []
for true_idx, pred_idx in zip(y_true, y_pred):
    if true_idx != pred_idx:
        misclassified.append((class_names[int(true_idx)], class_names[int(pred_idx)]))

confusion_pairs = Counter(misclassified).most_common(20)
confusion_df = pd.DataFrame(confusion_pairs, columns=["pair", "count"])
if not confusion_df.empty:
    confusion_df["true_class"] = confusion_df["pair"].apply(lambda x: x[0])
    confusion_df["predicted_class"] = confusion_df["pair"].apply(lambda x: x[1])
    confusion_df = confusion_df[["true_class", "predicted_class", "count"]]
    confusion_df.to_csv(os.path.join(RESULTS_DIR, "top_confusions.csv"), index=False)

# -----------------------------
# Save model card
# -----------------------------
with open(os.path.join(RESULTS_DIR, "model_card.md"), "w", encoding="utf-8") as f:
    f.write("# Calorify AI Model Card\n\n")
    f.write("## Model\n")
    f.write("- Architecture: ResNet50 classifier (custom head)\n")
    f.write(f"- Classes: {num_classes}\n\n")
    f.write("## Evaluation Summary\n")
    f.write(f"- Accuracy: {acc * 100:.2f}%\n")
    f.write(f"- Top-3 Accuracy: {top3_acc * 100:.2f}%\n")
    f.write(f"- Macro F1: {macro_f1:.4f}\n")
    f.write(f"- Weighted F1: {weighted_f1:.4f}\n")
    f.write(f"- ROC AUC (micro): {roc_auc_micro:.4f}\n")
    f.write(f"- PR AUC (micro): {pr_auc_micro:.4f}\n")
    f.write(f"- ECE (10 bins): {ece:.4f}\n")
    f.write(f"- Brier score: {brier:.4f}\n\n")
    f.write("## Known Limitations\n")
    f.write("- Performance may drop on mixed dishes, low-light images, or unusual presentation.\n")
    f.write("- Confidence is probabilistic, not guaranteed correctness.\n")
    f.write("- Portion and calorie estimation are approximate and should be user-adjusted.\n")
    f.write("- This system is decision support only, not medical advice.\n")

print("Classification report saved.")
print("Confusion matrix saved.")
print("Advanced metrics and model card saved.")
if confusion_pairs:
    print("Top confusion pairs:")
    for pair, count in confusion_pairs[:10]:
        print(f"{pair[0]} -> {pair[1]} : {count}")