import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

# -----------------------------
# Settings
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 40
WARMUP_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 6
IMG_SIZE = 224

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "food_model.pth"
LABEL_PATH = MODEL_DIR / "label_classes.txt"
HISTORY_JSON_PATH = MODEL_DIR / "training_history.json"
HISTORY_CSV_PATH = MODEL_DIR / "training_history.csv"
BEST_META_PATH = MODEL_DIR / "best_model_meta.json"

SEED = 42
WEIGHT_DECAY = 1e-4
HEAD_LR = 3e-4
BACKBONE_LR = 3e-5
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 0
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTS = True

DATASET_CSV = "dataset.csv"
TRAIN_SPLIT_CSV = "train_split.csv"
VAL_SPLIT_CSV = "val_split.csv"
TEST_SPLIT_CSV = "test_split.csv"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


def validate_dataset_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path)

    required_cols = {"image", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"dataset.csv is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["image", "label"]).copy()
    df["image"] = df["image"].astype(str)
    df["label"] = df["label"].astype(str)
    df = df[df["image"].apply(os.path.exists)].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found in dataset.csv after file existence filtering.")

    if df["label"].nunique() < 2:
        raise ValueError("Training requires at least 2 classes.")

    return df


# -----------------------------
# Load dataset
# -----------------------------
df = validate_dataset_csv(DATASET_CSV)

print("Total valid samples:", len(df))
print("Total classes:", df["label"].nunique())
print("Top 10 class counts:")
print(df["label"].value_counts().head(10))

# Encode labels
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

with open(LABEL_PATH, "w", encoding="utf-8") as f:
    for cls in label_encoder.classes_:
        f.write(cls + "\n")

# -----------------------------
# Split dataset
# -----------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label_id"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label_id"]
)

train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
val_df.to_csv(VAL_SPLIT_CSV, index=False)
test_df.to_csv(TEST_SPLIT_CSV, index=False)

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))


# -----------------------------
# Dataset class
# -----------------------------
class FoodDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["image"]
        label = int(row["label_id"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------------
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# -----------------------------
# Datasets / loaders
# -----------------------------
train_dataset = FoodDataset(train_df, transform=train_transform)
val_dataset = FoodDataset(val_df, transform=eval_transform)
test_dataset = FoodDataset(test_df, transform=eval_transform)

loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

train_loader = None
if USE_WEIGHTED_SAMPLER:
    class_counts = train_df["label_id"].value_counts().sort_index()
    sample_weights = train_df["label_id"].map(lambda x: 1.0 / class_counts.loc[x]).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
else:
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Model
# -----------------------------
num_classes = df["label_id"].nunique()

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

class_weights_tensor = None
if USE_CLASS_WEIGHTS:
    train_counts = train_df["label_id"].value_counts().sort_index()
    total = train_counts.sum()
    class_weights = total / (len(train_counts) * train_counts.values.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor,
    label_smoothing=LABEL_SMOOTHING
)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def make_optimizer_and_scheduler(model_ref, total_epochs: int):
    params = [
        {"params": model_ref.layer3.parameters(), "lr": BACKBONE_LR},
        {"params": model_ref.layer4.parameters(), "lr": BACKBONE_LR},
        {"params": model_ref.fc.parameters(), "lr": HEAD_LR},
    ]

    optimizer_ref = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    effective_epochs = max(1, total_epochs - WARMUP_EPOCHS)

    def cosine_decay(epoch_idx: int):
        progress = min(epoch_idx / effective_epochs, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler_ref = torch.optim.lr_scheduler.LambdaLR(
        optimizer_ref,
        lr_lambda=cosine_decay
    )
    return optimizer_ref, scheduler_ref


optimizer = None
scheduler = None


# -----------------------------
# Evaluation helper
# -----------------------------
def evaluate(loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / max(len(loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


# -----------------------------
# Training loop
# -----------------------------
best_val_acc = 0.0
best_epoch = -1
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}

for epoch in range(EPOCHS):
    if epoch == WARMUP_EPOCHS:
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        optimizer, scheduler = make_optimizer_and_scheduler(model, EPOCHS)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.fc.parameters(),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY
        )

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / max(len(train_loader), 1)
    train_acc = 100.0 * correct / max(total, 1)

    val_loss, val_acc = evaluate(val_loader)

    if scheduler is not None:
        scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

    history["train_loss"].append(round(float(train_loss), 6))
    history["val_loss"].append(round(float(val_loss), 6))
    history["train_acc"].append(round(float(train_acc), 6))
    history["val_acc"].append(round(float(val_acc), 6))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), MODEL_PATH)
        with open(BEST_META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_val_acc": round(float(best_val_acc), 6),
                    "num_classes": int(num_classes),
                    "seed": SEED,
                },
                f,
                indent=2,
            )
        print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after epoch {epoch + 1}.")
        break

print("Training complete.")
print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

pd.DataFrame(history).to_csv(HISTORY_CSV_PATH, index=False)
print(f"Training history saved to {HISTORY_JSON_PATH} and {HISTORY_CSV_PATH}")


# -----------------------------
# Final test evaluation
# -----------------------------
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
test_loss, test_acc = evaluate(test_loader)

print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Model saved to {MODEL_PATH}")
print(f"Labels saved to {LABEL_PATH}")