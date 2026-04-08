import os
from typing import List, Tuple

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(PROJECT_ROOT, "models", "label_classes.txt")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "food_model.pth")

IMG_SIZE = 224

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _validate_required_files() -> None:
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Missing label file: {LABELS_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


@st.cache_resource
def load_model_bundle():
    _validate_required_files()

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    if not class_names:
        raise ValueError("label_classes.txt is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(len(class_names))

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return class_names, model, device


def _predict_probs(image: Image.Image, use_tta: bool = True) -> torch.Tensor:
    class_names, model, device = load_model_bundle()

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
    class_names, _, _ = load_model_bundle()
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
    }


def heuristic_portion_grams(confidence_pct: float, calories_per_100g: float) -> float:
    """
    Rule-based portion estimate.
    Lower confidence -> slightly more conservative portion.
    Higher calorie density -> slightly smaller estimate.
    """
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