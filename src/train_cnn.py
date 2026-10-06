from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "train"
VALID_DIR = BASE_DIR / "data" / "valid"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "plots"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 20


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\n" + "=" * 60)
print("CLASS NAMES")
print("=" * 60)

print(class_names)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),

        tf.keras.layers.RandomRotation(0.05),

        tf.keras.layers.RandomZoom(0.10),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# BUILD CUSTOM CNN
# ============================================================

model = tf.keras.Sequential(
    [

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        tf.keras.layers.Input(
            shape=(224, 224, 3)
        ),

        # ----------------------------------------------------
        # Data Augmentation
        # ----------------------------------------------------

        data_augmentation,

        # ----------------------------------------------------
        # Normalize pixel values
        # ----------------------------------------------------

        tf.keras.layers.Rescaling(
            1.0 / 255
        ),

        # ----------------------------------------------------
        # Convolution Block 1
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Convolution Block 2
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Convolution Block 3
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Convolution Block 4
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            256,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        tf.keras.layers.Flatten(),

        # ----------------------------------------------------
        # Fully Connected Layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            256,
            activation="relu"
        ),

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        tf.keras.layers.Dropout(
            0.5
        ),

        # ----------------------------------------------------
        # Output Layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ],

    name="custom_cnn"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\n" + "=" * 60)
print("CUSTOM CNN MODEL")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_DIR / "custom_cnn.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING CNN TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "custom_cnn_final.keras"

model.save(final_model_path)

print("\nFinal model saved to:")
print(final_model_path)


# ============================================================
# PLOT TRAINING ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Custom CNN - Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

accuracy_path = (
    RESULTS_DIR /
    "custom_cnn_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Custom CNN - Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

loss_path = (
    RESULTS_DIR /
    "custom_cnn_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CNN TRAINING COMPLETED")
print("=" * 60)

print("\nBest model:")
print(MODEL_DIR / "custom_cnn.keras")

print("\nFinal model:")
print(final_model_path)

print("\nAccuracy plot:")
print(accuracy_path)

print("\nLoss plot:")
print(loss_path)

# ============================================================
# RESULTS SUMMARY
# ============================================================
#
# ## Custom CNN Training Results
#
# - Training images: 1,695
# - Validation images: 502
# - Test images: 246
# - Number of classes: 4
# - Input image size: 224 x 224 x 3
# - Batch size: 32
# - Maximum epochs: 20
#
# ### Best Result
#
# - Best validation accuracy: 84.46%
# - Best validation loss: 0.4824
# - Best epoch: 12
#
# ### Saved Files
#
# Best model:
# models/custom_cnn.keras
#
# Final model:
# models/custom_cnn_final.keras
#
# Accuracy plot:
# results/plots/custom_cnn_accuracy.png
#
# Loss plot:
# results/plots/custom_cnn_loss.png
#
# ### Observation
#
# Validation accuracy improved from 59.96% in Epoch 1
# to 84.46% in Epoch 12.
#
# After Epoch 12, validation accuracy became less stable
# while training accuracy continued to increase.
#
# Therefore, the checkpoint with the best validation accuracy
# will be used for test-set evaluation.
#
# ============================================================