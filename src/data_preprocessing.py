from pathlib import Path

import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"
VALID_DIR = BASE_DIR / "data" / "valid"
TEST_DIR = BASE_DIR / "data" / "test"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DISPLAY CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\n" + "=" * 60)
print("CLASS NAMES")
print("=" * 60)

print(class_names)


# ============================================================
# NORMALIZATION
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


# ============================================================
# PREPROCESS TRAINING DATA
# ============================================================

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# ============================================================
# PREPROCESS VALIDATION DATA
# ============================================================

valid_dataset = valid_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# ============================================================
# PREPROCESS TEST DATA
# ============================================================

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# ============================================================
# APPLY DATA AUGMENTATION TO TRAINING DATA ONLY
# ============================================================

train_dataset = train_dataset.map(
    lambda images, labels: (
        data_augmentation(images, training=True),
        labels
    )
)


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

valid_dataset = valid_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# CHECK ONE BATCH
# ============================================================

for images, labels in train_dataset.take(1):

    print("\n" + "=" * 60)
    print("PREPROCESSING + AUGMENTATION CHECK")
    print("=" * 60)

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print(
        "Minimum pixel value:",
        tf.reduce_min(images).numpy()
    )

    print(
        "Maximum pixel value:",
        tf.reduce_max(images).numpy()
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DATA PREPROCESSING AND AUGMENTATION COMPLETED")
print("=" * 60)