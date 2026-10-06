# ============================================================
# BRAIN TUMOR MRI CLASSIFICATION
# CUSTOM CNN - MODEL EVALUATION
# ============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# PATHS
# ============================================================

TEST_DIR = "data/test"
MODEL_PATH = "models/custom_cnn.keras"

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED
)

print("\nTest dataset loaded successfully.")
print("Class names:", test_dataset.class_names)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING CUSTOM CNN MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")
print("Model:", MODEL_PATH)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING MODEL")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    test_dataset,
    verbose=1
)

predicted_labels = np.argmax(predictions, axis=1)


# ============================================================
# GET TRUE LABELS
# ============================================================

true_labels = np.concatenate(
    [labels.numpy() for images, labels in test_dataset],
    axis=0
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print("\nAccuracy:", accuracy)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\n", cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Custom CNN - Confusion Matrix")

plt.tight_layout()

confusion_matrix_path = os.path.join(
    PLOTS_DIR,
    "custom_cnn_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved to:")
print(confusion_matrix_path)


# ============================================================
# SAVE RESULTS AS MARKDOWN
# ============================================================

markdown_path = os.path.join(
    RESULTS_DIR,
    "custom_cnn_test_results.md"
)

with open(markdown_path, "w", encoding="utf-8") as file:

    file.write("# Custom CNN Test Results\n\n")

    file.write("## Dataset\n\n")
    file.write("- Test images: 246\n")
    file.write("- Number of classes: 4\n")
    file.write("- Image size: 224 × 224\n")
    file.write("- Classes: glioma, meningioma, no_tumor, pituitary\n\n")

    file.write("## Model\n\n")
    file.write("- Model: Custom CNN\n")
    file.write("- Model file: `models/custom_cnn.keras`\n\n")

    file.write("## Test Performance\n\n")
    file.write(f"- **Test Accuracy:** {test_accuracy * 100:.2f}%\n")
    file.write(f"- **Test Loss:** {test_loss:.4f}\n\n")

    file.write("## Classification Report\n\n")
    file.write("```text\n")
    file.write(report)
    file.write("\n```\n\n")

    file.write("## Confusion Matrix\n\n")
    file.write(
        "The confusion matrix is saved at "
        "`results/plots/custom_cnn_confusion_matrix.png`.\n\n"
    )

    file.write("## Notes\n\n")
    file.write(
        "This model is intended for educational and research purposes "
        "and should not be used as a standalone medical diagnostic system.\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print("\nMarkdown results:")
print(markdown_path)

print("\nEvaluation completed successfully!")