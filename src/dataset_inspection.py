from pathlib import Path
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"
VALID_DIR = BASE_DIR / "data" / "valid"
TEST_DIR = BASE_DIR / "data" / "test"


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# DATASET INSPECTION FUNCTION
# ============================================================

def inspect_dataset(dataset_dir):
    """
    Inspect a dataset directory.

    Returns:
        dict: Number of images in each class.
    """

    print("\n" + "=" * 60)
    print(f"DATASET: {dataset_dir.name.upper()}")
    print("=" * 60)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset folder not found:")
        print(dataset_dir)
        return {}

    class_counts = {}
    total_images = 0

    # Get class folders
    class_folders = sorted(
        [folder for folder in dataset_dir.iterdir() if folder.is_dir()]
    )

    if not class_folders:
        print("No class folders found.")
        return {}

    print("\nClasses found:")

    for class_folder in class_folders:

        image_files = [
            file
            for file in class_folder.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        count = len(image_files)

        class_counts[class_folder.name] = count
        total_images += count

        print(f"  {class_folder.name}: {count} images")

    print("\nTotal images:", total_images)

    return class_counts


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("BRAIN TUMOR MRI DATASET INSPECTION")
    print("=" * 60)

    # Inspect each dataset
    train_counts = inspect_dataset(TRAIN_DIR)
    valid_counts = inspect_dataset(VALID_DIR)
    test_counts = inspect_dataset(TEST_DIR)

    # Final summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print("\nTrain:", train_counts)
    print("Validation:", valid_counts)
    print("Test:", test_counts)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETED")
    print("=" * 60)