import os
import pandas as pd

base_path = os.path.join("data", "raw", "food-101", "images")
output_file = "dataset.csv"

# Set to None to use all images
MAX_IMAGES_PER_CLASS = None

if not os.path.exists(base_path):
    raise FileNotFoundError(f"Dataset folder not found: {base_path}")

data = []

print(f"Processing classes from: {base_path}")

all_classes = [
    class_name for class_name in sorted(os.listdir(base_path))
    if os.path.isdir(os.path.join(base_path, class_name))
]

print(f"Folders found: {len(all_classes)}")

for class_name in all_classes:
    class_path = os.path.join(base_path, class_name)

    image_files = [
        f for f in sorted(os.listdir(class_path))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if MAX_IMAGES_PER_CLASS is not None:
        image_files = image_files[:MAX_IMAGES_PER_CLASS]

    print(f"{class_name}: {len(image_files)} images")

    for img_name in image_files:
        img_path = os.path.join(class_path, img_name)

        if os.path.isfile(img_path):
            data.append({
                "image": img_path,
                "label": class_name
            })

df = pd.DataFrame(data)

# Final safety check: keep only rows whose files still exist
df = df[df["image"].apply(os.path.exists)].reset_index(drop=True)

df.to_csv(output_file, index=False)

print("Dataset created successfully.")
print("Total samples:", len(df))
print("Total classes in CSV:", df["label"].nunique())
print("First 10 classes:", sorted(df["label"].unique())[:10])
print(df.head())