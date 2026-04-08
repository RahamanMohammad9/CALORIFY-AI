import os
from pathlib import Path

import pandas as pd

BASE_PATH = Path("data") / "raw" / "food-101" / "images"
CSV_PATH = "dataset.csv"

if not BASE_PATH.exists():
    raise FileNotFoundError(f"Dataset folder not found: {BASE_PATH}")

real_count = 0
class_counts = {}

for class_name in sorted(os.listdir(BASE_PATH)):
    class_path = BASE_PATH / class_name

    if not class_path.is_dir():
        continue

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    class_counts[class_name] = len(image_files)
    real_count += len(image_files)

print("REAL IMAGE COUNT:", real_count)
print("REAL CLASS COUNT:", len(class_counts))

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Missing CSV file: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_cols = {"image", "label"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"dataset.csv missing columns: {sorted(missing_cols)}")

print("CSV ROW COUNT:", len(df))
print("CSV CLASS COUNT:", df["label"].nunique())

missing = df[~df["image"].apply(os.path.exists)]
print("MISSING FILE PATHS INSIDE CSV:", len(missing))
if len(missing) > 0:
    print(missing.head())

duplicate_paths = df[df.duplicated(subset=["image"], keep=False)]
print("DUPLICATE IMAGE PATHS IN CSV:", len(duplicate_paths))

csv_class_counts = df["label"].value_counts().to_dict()

comparison_rows = []
for cls, real_cls_count in class_counts.items():
    csv_cls_count = csv_class_counts.get(cls, 0)
    comparison_rows.append({
        "class": cls,
        "real_count": real_cls_count,
        "csv_count": csv_cls_count,
        "difference": real_cls_count - csv_cls_count,
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\nCLASS COUNT COMPARISON:")
print(comparison_df.head(20))

mismatch_df = comparison_df[comparison_df["difference"] != 0]
print("\nMISMATCHED CLASSES:", len(mismatch_df))
if not mismatch_df.empty:
    print(mismatch_df.head(20))