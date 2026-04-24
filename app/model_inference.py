import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "models" / "label_classes.txt"
RESULTS_PATH = PROJECT_ROOT / "results" / "model_comparison_results.csv"

IMG_SIZE = 224

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_best_model_path() -> tuple[Path, str]:
    if not RESULTS_PATH.exists():
        fallback_model = PROJECT_ROOT / "models" / "food_model.pth"
        if fallback_model.exists():
            return fallback_model, "resnet50"
        raise FileNotFoundError(
            f"Missing comparison file: {RESULTS_PATH} and fallback model not found."
        )

    df = pd.read_csv(RESULTS_PATH)

    if "model_key" not in df.columns or "test_accuracy_pct" not in df.columns:
        raise ValueError(
            "model_comparison_results.csv must contain 'model_key' and 'test_accuracy_pct' columns."
        )

    best_row = df.sort_values("test_accuracy_pct", ascending=False).iloc[0]
    best_model_key = str(best_row["model_key"]).strip()
    model_path = PROJECT_ROOT / "models" / f"{best_model_key}_best.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Best model file not found: {model_path}")

    return model_path, best_model_key


def _validate_required_files(model_path: Path) -> None:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing label file: {LABELS_PATH}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")


def _build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    if model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


@st.cache_resource
def load_model_bundle():
    model_path, model_name = get_best_model_path()
    _validate_required_files(model_path)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    if not class_names:
        raise ValueError("label_classes.txt is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_name, len(class_names))

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return class_names, model, device, model_name


def _predict_probs(image: Image.Image, use_tta: bool = True) -> torch.Tensor:
    class_names, model, device, _ = load_model_bundle()

    if image.mode != "RGB":
        image = image.convert("RGB")

    with torch.no_grad():
        x = _INFER_TRANSFORM(image).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        if use_tta:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            x_flip = _INFER_TRANSFORM(flipped).unsqueeze(0).to(device)
            logits_flip = model(x_flip)
            probs_flip = torch.softmax(logits_flip, dim=1)
            probs = (probs + probs_flip) / 2.0

    if probs.shape[1] != len(class_names):
        raise ValueError(
            f"Prediction output size ({probs.shape[1]}) does not match label count ({len(class_names)})."
        )

    return probs


def predict_topk(
    image: Image.Image,
    top_k: int = 3,
    use_tta: bool = True
) -> List[Tuple[str, float]]:
    class_names, _, _, _ = load_model_bundle()
    probs = _predict_probs(image, use_tta=use_tta)

    safe_top_k = max(1, min(int(top_k), len(class_names)))
    top_prob, top_idx = torch.topk(probs, k=safe_top_k, dim=1)

    results: List[Tuple[str, float]] = []
    for p, i in zip(top_prob[0], top_idx[0]):
        results.append((class_names[int(i)], float(p) * 100.0))

    return results


def predict_with_confidence_band(
    image: Image.Image,
    top_k: int = 3,
    use_tta: bool = True
) -> dict:
    predictions = predict_topk(image=image, top_k=top_k, use_tta=use_tta)
    best_label, best_conf = predictions[0]
    _, _, _, model_name = load_model_bundle()

    if best_conf >= 85:
        band = "high"
        explanation = "The model is strongly confident in this prediction."
    elif best_conf >= 65:
        band = "medium"
        explanation = "The model is moderately confident. User confirmation is still helpful."
    else:
        band = "low"
        explanation = "The model confidence is low. The prediction should be treated carefully."

    return {
        "label": best_label,
        "confidence": best_conf,
        "confidence_band": band,
        "explanation": explanation,
        "top_predictions": predictions,
        "model_name": model_name,
    }


def heuristic_portion_grams(confidence_pct: float, calories_per_100g: float) -> float:
    confidence_pct = float(confidence_pct)
    calories_per_100g = float(calories_per_100g)

    if confidence_pct < 60:
        conf_factor = 0.85
    elif confidence_pct < 80:
        conf_factor = 0.95
    else:
        conf_factor = 1.05

    if calories_per_100g > 280:
        density_factor = 0.9
    elif calories_per_100g < 140:
        density_factor = 1.1
    else:
        density_factor = 1.0

    grams = 120.0 * conf_factor * density_factor
    return max(40.0, min(300.0, grams))