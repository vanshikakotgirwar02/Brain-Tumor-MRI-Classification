# ============================================================
# BRAIN TUMOR MRI CLASSIFICATION
# TRANSFER LEARNING - MOBILENETV2
# ============================================================

import os
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = "data/train"
VALID_DIR = "data/valid"

MODEL_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 15

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTraining classes:", train_dataset.class_names)
print("Validation classes:", valid_dataset.class_names)


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
        )
    ],
    name="data_augmentation"
)


# ============================================================
# LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\n" + "=" * 70)
print("LOADING MOBILENETV2")
print("=" * 70)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

print("\nMobileNetV2 loaded successfully.")
print("Pretrained layers are frozen.")


# ============================================================
# BUILD TRANSFER LEARNING MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.4)(x)

outputs = tf.keras.layers.Dense(
    4,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="MobileNetV2_TransferLearning"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_transfer.keras"
)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True
    )
]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRANSFER LEARNING TRAINING")
print("=" * 70)

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_transfer_final.keras"
)

model.save(final_model_path)

print("\nFinal model saved to:")
print(final_model_path)


# ============================================================
# SAVE ACCURACY PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Transfer Learning - Accuracy")

plt.legend()
plt.tight_layout()

accuracy_plot_path = os.path.join(
    PLOTS_DIR,
    "mobilenetv2_accuracy.png"
)

plt.savefig(
    accuracy_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE LOSS PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 Transfer Learning - Loss")

plt.legend()
plt.tight_layout()

loss_plot_path = os.path.join(
    PLOTS_DIR,
    "mobilenetv2_loss.png"
)

plt.savefig(
    loss_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TRAINING RESULTS
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)

best_accuracy_epoch = (
    history.history["val_accuracy"].index(
        best_val_accuracy
    ) + 1
)

best_loss_epoch = (
    history.history["val_loss"].index(
        best_val_loss
    ) + 1
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TRANSFER LEARNING RESULTS")
print("=" * 70)

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy Epoch: "
    f"{best_accuracy_epoch}"
)

print(
    f"\nLowest Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Lowest Validation Loss Epoch: "
    f"{best_loss_epoch}"
)


# ============================================================
# SAVE MARKDOWN RESULTS
# ============================================================

markdown_path = os.path.join(
    RESULTS_DIR,
    "mobilenetv2_training_results.md"
)

with open(
    markdown_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("# MobileNetV2 Transfer Learning Results\n\n")

    file.write("## Dataset\n\n")
    file.write("- Training images: 1,695\n")
    file.write("- Validation images: 502\n")
    file.write("- Number of classes: 4\n")
    file.write("- Image size: 224 × 224\n\n")

    file.write("## Model\n\n")
    file.write("- Architecture: MobileNetV2\n")
    file.write("- Transfer learning: ImageNet pretrained weights\n")
    file.write("- Pretrained layers: Frozen\n")
    file.write("- Optimizer: Adam\n")
    file.write("- Learning rate: 0.0001\n\n")

    file.write("## Results\n\n")

    file.write(
        f"- **Best Validation Accuracy:** "
        f"{best_val_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"- **Best Accuracy Epoch:** "
        f"{best_accuracy_epoch}\n"
    )

    file.write(
        f"- **Lowest Validation Loss:** "
        f"{best_val_loss:.4f}\n"
    )

    file.write(
        f"- **Lowest Loss Epoch:** "
        f"{best_loss_epoch}\n"
    )

    file.write("\n## Saved Files\n\n")

    file.write(
        "- `models/mobilenetv2_transfer.keras`\n"
    )

    file.write(
        "- `models/mobilenetv2_transfer_final.keras`\n"
    )

    file.write(
        "- `results/plots/mobilenetv2_accuracy.png`\n"
    )

    file.write(
        "- `results/plots/mobilenetv2_loss.png`\n"
    )


# ============================================================
# MARKDOWN-STYLE RESULTS SUMMARY
# ============================================================
#
# ## MobileNetV2 Transfer Learning
#
# The model uses ImageNet pretrained MobileNetV2
# with frozen convolutional base layers.
#
# Training and validation results are automatically
# calculated and saved to:
#
# `results/mobilenetv2_training_results.md`
#
# The actual accuracy and loss values should be taken
# from the generated Markdown results file.
#
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nMarkdown results saved to:")
print(markdown_path)

print("\nAccuracy plot saved to:")
print(accuracy_plot_path)

print("\nLoss plot saved to:")
print(loss_plot_path)