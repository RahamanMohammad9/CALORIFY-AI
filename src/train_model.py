import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# -----------------------------
# Settings (tuned for higher accuracy)
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 35
WARMUP_EPOCHS = 3
IMG_SIZE = 224
MODEL_PATH = "models/food_model.pth"
LABEL_PATH = "models/label_classes.txt"
HISTORY_JSON_PATH = "models/training_history.json"
HISTORY_CSV_PATH = "models/training_history.csv"
SEED = 42
WEIGHT_DECAY = 1e-4
HEAD_LR = 3e-4
BACKBONE_LR = 3e-5
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 0

os.makedirs("models", exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("dataset.csv")
df = df[df["image"].apply(os.path.exists)].reset_index(drop=True)

print("Total valid samples:", len(df))
print("Total classes:", df["label"].nunique())

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
    random_state=42,
    stratify=df["label_id"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label_id"]
)

train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

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
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# -----------------------------
# Loaders
# -----------------------------
train_dataset = FoodDataset(train_df, transform=train_transform)
val_dataset = FoodDataset(val_df, transform=eval_transform)
test_dataset = FoodDataset(test_df, transform=eval_transform)

loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

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

# Warmup stage: train head only.
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

# Unfreeze classifier head
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


def make_optimizer_and_scheduler(model_ref, total_epochs: int):
    params = [
        {"params": model_ref.layer3.parameters(), "lr": BACKBONE_LR},
        {"params": model_ref.layer4.parameters(), "lr": BACKBONE_LR},
        {"params": model_ref.fc.parameters(), "lr": HEAD_LR},
    ]
    opt = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)

    effective_epochs = max(1, total_epochs - WARMUP_EPOCHS)

    def cosine_decay(epoch_idx: int):
        progress = min(epoch_idx / effective_epochs, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_decay)
    return opt, sch


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

    avg_loss = running_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

# -----------------------------
# Training loop
# -----------------------------
best_val_acc = 0.0
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}

for epoch in range(EPOCHS):
    if epoch == WARMUP_EPOCHS:
        # Fine-tune deeper layers after head warmup.
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        optimizer, scheduler = make_optimizer_and_scheduler(model, EPOCHS)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
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

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    val_loss, val_acc = evaluate(val_loader)
    if scheduler is not None:
        scheduler.step()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

    history["train_loss"].append(round(float(train_loss), 6))
    history["val_loss"].append(round(float(val_loss), 6))
    history["train_acc"].append(round(float(train_acc), 6))
    history["val_acc"].append(round(float(val_acc), 6))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")

print("Training complete.")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

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