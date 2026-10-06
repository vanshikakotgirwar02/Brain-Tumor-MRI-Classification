from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"


# ============================================================
# SETTINGS
# ============================================================

IMAGES_PER_CLASS = 4

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# GET CLASS FOLDERS
# ============================================================

class_folders = sorted(
    [
        folder
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    ]
)

print("\nClasses found:")

for folder in class_folders:
    print(f"  - {folder.name}")


# ============================================================
# CREATE VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    len(class_folders),
    IMAGES_PER_CLASS,
    figsize=(12, 12)
)


# ============================================================
# DISPLAY IMAGES
# ============================================================

for row, class_folder in enumerate(class_folders):

    image_files = [
        file
        for file in class_folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Randomly select images
    selected_images = random.sample(
        image_files,
        min(IMAGES_PER_CLASS, len(image_files))
    )

    for col in range(IMAGES_PER_CLASS):

        ax = axes[row, col]

        if col < len(selected_images):

            image_path = selected_images[col]

            image = Image.open(image_path)

            ax.imshow(image, cmap="gray")

            ax.set_title(
                class_folder.name,
                fontsize=10
            )

        ax.axis("off")


# ============================================================
# FINALIZE PLOT
# ============================================================

plt.suptitle(
    "Brain Tumor MRI Dataset - Sample Images",
    fontsize=16
)

plt.tight_layout()

# ============================================================
# SAVE VISUALIZATION
# ============================================================

RESULTS_DIR = BASE_DIR / "results" / "plots"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = RESULTS_DIR / "sample_mri_images.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nVisualization saved to:")
print(output_path)

plt.show()