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

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model_bundle():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(class_names)),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return class_names, model, device


def predict_topk(image: Image.Image, top_k: int = 3, use_tta: bool = True) -> List[Tuple[str, float]]:
    class_names, model, device = load_model_bundle()
    with torch.no_grad():
        x = _TRANSFORM(image).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        if use_tta:
            x_flip = _TRANSFORM(image.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)
            logits_flip = model(x_flip)
            probs_flip = torch.softmax(logits_flip, dim=1)
            probs = (probs + probs_flip) / 2.0
        top_prob, top_idx = torch.topk(probs, k=top_k, dim=1)
    return [(class_names[int(i)], float(p) * 100.0) for p, i in zip(top_prob[0], top_idx[0])]


def heuristic_portion_grams(confidence_pct: float, calories_per_100g: float) -> float:
    # Basic production-safe heuristic: lower confidence => more conservative estimate.
    conf_factor = 0.85 if confidence_pct < 60 else (0.95 if confidence_pct < 80 else 1.05)
    density_factor = 0.9 if calories_per_100g > 280 else (1.1 if calories_per_100g < 140 else 1.0)
    grams = 120.0 * conf_factor * density_factor
    return max(40.0, min(300.0, grams))
