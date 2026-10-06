from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"

RESULTS_DIR = BASE_DIR / "results" / "plots"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# COUNT IMAGES IN EACH CLASS
# ============================================================

class_counts = {}

for class_folder in sorted(TRAIN_DIR.iterdir()):

    if not class_folder.is_dir():
        continue

    image_files = [
        file
        for file in class_folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    class_counts[class_folder.name] = len(image_files)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING DATASET CLASS DISTRIBUTION")
print("=" * 60)

for class_name, count in class_counts.items():

    print(f"{class_name}: {count} images")


print(f"\nTotal images: {sum(class_counts.values())}")


# ============================================================
# CREATE BAR CHART
# ============================================================

classes = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(10, 6))

plt.bar(classes, counts)

plt.title(
    "Brain Tumor MRI Dataset - Class Distribution",
    fontsize=16
)

plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=0)

plt.tight_layout()


# ============================================================
# SAVE CHART
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = RESULTS_DIR / "class_distribution.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nVisualization saved to:")
print(output_path)


# ============================================================
# DISPLAY CHART
# ============================================================

plt.show()