from pathlib import Path
from torchvision.datasets import Food101


def ensure_food101():
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"

    # This is where Food101 stores files
    dataset_folder = data_root / "food-101"
    images_folder = dataset_folder / "images"

    # ✅ STRONG CHECK
    if dataset_folder.exists() and images_folder.exists():
        # Check if images folder is not empty
        if any(images_folder.iterdir()):
            print("Food-101 already exists. Skipping download.")
            return

    print("Food-101 not found. Downloading...")
    Food101(root=str(data_root), split="train", download=True)
    Food101(root=str(data_root), split="test", download=True)
    print("Food-101 download complete.")