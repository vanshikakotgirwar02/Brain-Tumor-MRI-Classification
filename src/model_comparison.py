# ============================================================
# BRAIN TUMOR MRI CLASSIFICATION
# MODEL COMPARISON
# CUSTOM CNN vs MOBILENETV2
# ============================================================

import os
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# MODEL RESULTS
# ============================================================

custom_cnn_accuracy = 81.71
custom_cnn_loss = 0.4954
custom_cnn_macro_f1 = 81.05
custom_cnn_weighted_f1 = 81.48

mobilenet_accuracy = 80.89
mobilenet_loss = 0.5010
mobilenet_macro_f1 = 80.45
mobilenet_weighted_f1 = 80.56


# ============================================================
# PRINT COMPARISON
# ============================================================

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("\nCustom CNN")
print("-" * 30)
print(f"Test Accuracy : {custom_cnn_accuracy:.2f}%")
print(f"Test Loss     : {custom_cnn_loss:.4f}")
print(f"Macro F1      : {custom_cnn_macro_f1:.2f}%")
print(f"Weighted F1   : {custom_cnn_weighted_f1:.2f}%")

print("\nMobileNetV2")
print("-" * 30)
print(f"Test Accuracy : {mobilenet_accuracy:.2f}%")
print(f"Test Loss     : {mobilenet_loss:.4f}")
print(f"Macro F1      : {mobilenet_macro_f1:.2f}%")
print(f"Weighted F1   : {mobilenet_weighted_f1:.2f}%")


# ============================================================
# DETERMINE BEST MODEL
# ============================================================

if custom_cnn_accuracy > mobilenet_accuracy:
    best_model = "Custom CNN"
else:
    best_model = "MobileNetV2"


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"\nBest model based on test accuracy: {best_model}")


# ============================================================
# CREATE COMPARISON CHART
# ============================================================

models = [
    "Custom CNN",
    "MobileNetV2"
]

accuracies = [
    custom_cnn_accuracy,
    mobilenet_accuracy
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    accuracies
)

plt.ylabel("Test Accuracy (%)")
plt.xlabel("Model")
plt.title("Custom CNN vs MobileNetV2")

plt.ylim(0, 100)

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()

comparison_plot_path = os.path.join(
    PLOTS_DIR,
    "model_accuracy_comparison.png"
)

plt.savefig(
    comparison_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison chart saved to:")
print(comparison_plot_path)


# ============================================================
# SAVE MARKDOWN REPORT
# ============================================================

markdown_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.md"
)

with open(
    markdown_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("# Brain Tumor MRI Classification - Model Comparison\n\n")

    file.write(
        "This report compares the performance of a Custom CNN "
        "and MobileNetV2 Transfer Learning model.\n\n"
    )

    file.write("## Model Performance\n\n")

    file.write(
        "| Metric | Custom CNN | MobileNetV2 |\n"
    )

    file.write(
        "|---|---:|---:|\n"
    )

    file.write(
        f"| Test Accuracy | {custom_cnn_accuracy:.2f}% | "
        f"{mobilenet_accuracy:.2f}% |\n"
    )

    file.write(
        f"| Test Loss | {custom_cnn_loss:.4f} | "
        f"{mobilenet_loss:.4f} |\n"
    )

    file.write(
        f"| Macro F1 | {custom_cnn_macro_f1:.2f}% | "
        f"{mobilenet_macro_f1:.2f}% |\n"
    )

    file.write(
        f"| Weighted F1 | {custom_cnn_weighted_f1:.2f}% | "
        f"{mobilenet_weighted_f1:.2f}% |\n\n"
    )

    file.write("## Best Model\n\n")

    file.write(
        f"**{best_model}** achieved the higher test accuracy "
        f"and is selected as the better-performing model "
        f"for this dataset.\n\n"
    )

    file.write("## Analysis\n\n")

    file.write(
        "- Custom CNN achieved slightly higher overall test "
        "accuracy.\n"
    )

    file.write(
        "- MobileNetV2 used significantly fewer trainable "
        "parameters.\n"
    )

    file.write(
        "- MobileNetV2 performed particularly well on the "
        "no_tumor class.\n"
    )

    file.write(
        "- Custom CNN performed particularly well on the "
        "pituitary class.\n"
    )

    file.write(
        "- Meningioma was a challenging class for both models.\n\n"
    )

    file.write("## Conclusion\n\n")

    file.write(
        "Based on test accuracy and overall F1 performance, "
        "the Custom CNN performed slightly better on this "
        "dataset. MobileNetV2 provides a more lightweight "
        "alternative with substantially fewer parameters.\n\n"
    )

    file.write(
        "The selected model should be treated as an "
        "educational and research model and not as a "
        "standalone medical diagnostic system.\n"
    )

    file.write("\n## Comparison Chart\n\n")

    file.write(
        "Saved at: "
        "`results/plots/model_accuracy_comparison.png`\n"
    )


# ============================================================
# MARKDOWN-STYLE RESULTS SUMMARY
# ============================================================
#
# ## Model Comparison Results
#
# ### Custom CNN
# - Test Accuracy: 81.71%
# - Test Loss: 0.4954
# - Macro F1: 81.05%
# - Weighted F1: 81.48%
#
# ### MobileNetV2
# - Test Accuracy: 80.89%
# - Test Loss: 0.5010
# - Macro F1: 80.45%
# - Weighted F1: 80.56%
#
# ### Best Model
# Custom CNN
#
# ============================================================


print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)

print("\nMarkdown report saved to:")
print(markdown_path)

print("\nComparison chart saved to:")
print(comparison_plot_path)