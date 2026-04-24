import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 48
EPOCHS = 20
WARMUP_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0

WEIGHT_DECAY = 1e-4
HEAD_LR = 3e-4
BACKBONE_LR = 3e-5
LABEL_SMOOTHING = 0.1

USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTS = True

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_CSV = BASE_DIR / "dataset.csv"
TRAIN_SPLIT_CSV = BASE_DIR / "train_split.csv"
VAL_SPLIT_CSV = BASE_DIR / "val_split.csv"
TEST_SPLIT_CSV = BASE_DIR / "test_split.csv"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

COMPARISON_JSON = RESULTS_DIR / "model_comparison_results.json"
COMPARISON_CSV = RESULTS_DIR / "model_comparison_results.csv"

MODEL_CONFIGS = {
    "resnet50": {"display_name": "ResNet50"},
    "resnet18": {"display_name": "ResNet18"},
    "mobilenet_v3_large": {"display_name": "MobileNetV3-Large"},
    "efficientnet_b0": {"display_name": "EfficientNet-B0"},
    "densenet121": {"display_name": "DenseNet121"},
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def validate_dataset_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
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
        raise ValueError("No valid rows found in dataset.csv after filtering.")
    if df["label"].nunique() < 2:
        raise ValueError("Training requires at least 2 classes.")

    return df


class FoodDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        img_path = row["image"]
        label = int(row["label_id"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def create_data_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_dataset = FoodDataset(train_df, transform=get_train_transform())
    val_dataset = FoodDataset(val_df, transform=get_eval_transform())
    test_dataset = FoodDataset(test_df, transform=get_eval_transform())

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

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
    return train_loader, val_loader, test_loader


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
        for param in model.fc.parameters():
            param.requires_grad = True
        return model

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
        for param in model.fc.parameters():
            param.requires_grad = True
        return model

    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def unfreeze_for_finetune(model_name: str, model: nn.Module) -> None:
    if model_name == "resnet50":
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        return

    if model_name == "resnet18":
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        return

    if model_name == "mobilenet_v3_large":
        for param in model.features[-4:].parameters():
            param.requires_grad = True
        return

    if model_name == "efficientnet_b0":
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        return

    if model_name == "densenet121":
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True
        return


def make_optimizer_and_scheduler(model: nn.Module, total_epochs: int):
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": BACKBONE_LR})
    if head_params:
        param_groups.append({"params": head_params, "lr": HEAD_LR})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    effective_epochs = max(1, total_epochs - WARMUP_EPOCHS)

    def cosine_decay(epoch_idx: int):
        progress = min(epoch_idx / effective_epochs, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_decay)
    return optimizer, scheduler


def evaluate_loss_acc(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = running_loss / max(len(loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def evaluate_full_metrics(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    top3_acc = top_k_accuracy_score(
        y_true,
        y_prob,
        k=min(3, y_prob.shape[1]),
        labels=list(range(y_prob.shape[1])),
    )

    return {
        "test_accuracy_pct": round(acc * 100.0, 4),
        "test_macro_f1": round(float(macro_f1), 6),
        "test_weighted_f1": round(float(weighted_f1), 6),
        "test_top3_accuracy_pct": round(top3_acc * 100.0, 4),
    }


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model_path: Path) -> float:
    if not model_path.exists():
        return 0.0
    return round(model_path.stat().st_size / (1024 * 1024), 3)


def benchmark_inference_ms(model: nn.Module, loader: DataLoader, device: torch.device, batches: int = 5) -> float:
    model.eval()
    timings = []

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= batches:
                break

            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0 / max(images.size(0), 1)
            timings.append(elapsed_ms)

    return round(float(np.mean(timings)), 4) if timings else 0.0


def train_one_model(
    model_key: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_df: pd.DataFrame,
    num_classes: int,
    device: torch.device,
):
    print("\n" + "=" * 70)
    print(f"Training model: {MODEL_CONFIGS[model_key]['display_name']}")
    print("=" * 70)

    model = build_model(model_key, num_classes).to(device)

    class_weights_tensor = None
    if USE_CLASS_WEIGHTS:
        train_counts = train_df["label_id"].value_counts().sort_index()
        total = train_counts.sum()
        class_weights = total / (len(train_counts) * train_counts.values.astype(np.float32))
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=LABEL_SMOOTHING,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    optimizer = None
    scheduler = None

    best_val_acc = 0.0
    best_epoch = -1
    epochs_without_improvement = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    model_path = MODELS_DIR / f"{model_key}_best.pth"
    history_path = MODELS_DIR / f"{model_key}_history.csv"

    train_start = time.perf_counter()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for epoch in range(EPOCHS):
        if epoch == WARMUP_EPOCHS:
            unfreeze_for_finetune(model_key, model)
            optimizer, scheduler = make_optimizer_and_scheduler(model, EPOCHS)

        if optimizer is None:
            head_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(head_params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = 100.0 * correct / max(total, 1)
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{model_key} | Epoch [{epoch + 1}/{EPOCHS}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(round(float(train_loss), 6))
        history["val_loss"].append(round(float(val_loss), 6))
        history["train_acc"].append(round(float(train_acc), 6))
        history["val_acc"].append(round(float(val_acc), 6))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved: {model_path} | Val Acc: {best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered for {model_key} at epoch {epoch + 1}.")
            break

    training_time_sec = time.perf_counter() - train_start

    pd.DataFrame(history).to_csv(history_path, index=False)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_metrics = evaluate_full_metrics(model, test_loader, device)
    inference_ms_per_image = benchmark_inference_ms(model, test_loader, device)

    result = {
        "model_key": model_key,
        "model_name": MODEL_CONFIGS[model_key]["display_name"],
        "best_epoch": best_epoch,
        "best_val_accuracy_pct": round(float(best_val_acc), 4),
        "training_time_sec": round(float(training_time_sec), 2),
        "trainable_params": int(count_trainable_params(model)),
        "total_params": int(count_total_params(model)),
        "saved_model_size_mb": estimate_model_size_mb(model_path),
        "avg_inference_ms_per_image": inference_ms_per_image,
        **test_metrics,
    }

    print(f"\nFinished {result['model_name']}")
    print(json.dumps(result, indent=2))
    return result


def main():
    set_seed(SEED)

    df = validate_dataset_csv(DATASET_CSV)

    print("Total valid samples:", len(df))
    print("Total classes:", df["label"].nunique())

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])

    labels_path = MODELS_DIR / "label_classes.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for cls in label_encoder.classes_:
            f.write(cls + "\n")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label_id"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label_id"],
    )

    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(VAL_SPLIT_CSV, index=False)
    test_df.to_csv(TEST_SPLIT_CSV, index=False)

    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))

    train_loader, val_loader, test_loader = create_data_loaders(train_df, val_df, test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = df["label_id"].nunique()
    all_results = []

    for model_key in MODEL_CONFIGS.keys():
        result = train_one_model(
            model_key=model_key,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_df=train_df,
            num_classes=num_classes,
            device=device,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        by=["test_accuracy_pct", "test_macro_f1"],
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(COMPARISON_CSV, index=False)

    with open(COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison CSV to: {COMPARISON_CSV}")
    print(f"Saved comparison JSON to: {COMPARISON_JSON}")


if __name__ == "__main__":
    main()