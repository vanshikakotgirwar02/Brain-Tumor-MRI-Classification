from pathlib import Path
from collections import Counter
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIRS = {
    "train": BASE_DIR / "data" / "train",
    "valid": BASE_DIR / "data" / "valid",
    "test": BASE_DIR / "data" / "test"
}


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# IMAGE QUALITY CHECK FUNCTION
# ============================================================

def check_dataset(dataset_name, dataset_dir):

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} IMAGE QUALITY CHECK")
    print("=" * 70)

    dimension_counter = Counter()

    total_images = 0
    valid_images = 0
    corrupted_images = []

    # --------------------------------------------------------
    # Loop through class folders
    # --------------------------------------------------------

    for class_folder in sorted(dataset_dir.iterdir()):

        if not class_folder.is_dir():
            continue

        print(f"\nClass: {class_folder.name}")

        class_total = 0
        class_valid = 0

        for image_path in class_folder.iterdir():

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total_images += 1
            class_total += 1

            try:

                with Image.open(image_path) as image:

                    # Verify image integrity
                    image.verify()

                # Open again because verify() closes the image
                with Image.open(image_path) as image:

                    width, height = image.size

                    dimension_counter[(width, height)] += 1

                valid_images += 1
                class_valid += 1

            except Exception as error:

                corrupted_images.append(
                    {
                        "file": str(image_path),
                        "error": str(error)
                    }
                )

        print(f"  Total images: {class_total}")
        print(f"  Valid images: {class_valid}")

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(f"Total images: {total_images}")
    print(f"Valid images: {valid_images}")
    print(f"Corrupted images: {len(corrupted_images)}")

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    print("\nImage Dimensions:")

    for dimension, count in dimension_counter.most_common():

        width, height = dimension

        print(
            f"  {width} x {height}: {count} images"
        )

    # --------------------------------------------------------
    # Corrupted images
    # --------------------------------------------------------

    if corrupted_images:

        print("\nCorrupted Images:")

        for item in corrupted_images:

            print(f"\nFile: {item['file']}")
            print(f"Error: {item['error']}")

    else:

        print("\nNo corrupted images found. ✓")

    return dimension_counter, corrupted_images


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("BRAIN TUMOR MRI IMAGE QUALITY ANALYSIS")
    print("=" * 70)

    all_results = {}

    for dataset_name, dataset_dir in DATASET_DIRS.items():

        dimensions, corrupted = check_dataset(
            dataset_name,
            dataset_dir
        )

        all_results[dataset_name] = {
            "dimensions": dimensions,
            "corrupted": corrupted
        }

    print("\n" + "=" * 70)
    print("IMAGE QUALITY CHECK COMPLETED")
    print("=" * 70)