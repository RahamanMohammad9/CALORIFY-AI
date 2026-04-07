import os
import pandas as pd

base_path = os.path.join("data", "raw", "food-101", "images")

real_count = 0
class_counts = {}

for class_name in sorted(os.listdir(base_path)):
    class_path = os.path.join(base_path, class_name)

    if not os.path.isdir(class_path):
        continue

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    class_counts[class_name] = len(image_files)
    real_count += len(image_files)

print("REAL IMAGE COUNT:", real_count)
print("REAL CLASS COUNT:", len(class_counts))

df = pd.read_csv("dataset.csv")
print("CSV ROW COUNT:", len(df))
print("CSV CLASS COUNT:", df["label"].nunique())

# check missing files from csv
missing = df[~df["image"].apply(os.path.exists)]
print("MISSING FILE PATHS INSIDE CSV:", len(missing))

if len(missing) > 0:
    print(missing.head())