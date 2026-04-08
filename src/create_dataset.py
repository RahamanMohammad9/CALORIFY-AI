import os
from pathlib import Path

import pandas as pd

BASE_PATH = Path("data") / "raw" / "food-101" / "images"
OUTPUT_FILE = "dataset.csv"

# Set to None to use all images
MAX_IMAGES_PER_CLASS = None

if not BASE_PATH.exists():
    raise FileNotFoundError(f"Dataset folder not found: {BASE_PATH}")

data = []

print(f"Processing classes from: {BASE_PATH}")

all_classes = sorted([
    class_name for class_name in os.listdir(BASE_PATH)
    if (BASE_PATH / class_name).is_dir()
])

print(f"Folders found: {len(all_classes)}")

for class_name in all_classes:
    class_path = BASE_PATH / class_name

    image_files = sorted([
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if MAX_IMAGES_PER_CLASS is not None:
        image_files = image_files[:MAX_IMAGES_PER_CLASS]

    print(f"{class_name}: {len(image_files)} images")

    for img_name in image_files:
        img_path = class_path / img_name
        if img_path.is_file():
            data.append({
                "image": str(img_path),
                "label": class_name
            })

df = pd.DataFrame(data)

if df.empty:
    raise ValueError("No image rows were collected. Check dataset path and file extensions.")

df = df[df["image"].apply(os.path.exists)].reset_index(drop=True)
df = df.drop_duplicates(subset=["image"]).reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset created successfully.")
print("Total samples:", len(df))
print("Total classes in CSV:", df["label"].nunique())
print("Class distribution summary:")
print(df["label"].value_counts().describe())
print(df.head())