# Brain Tumor MRI Classification - Model Comparison

This report compares the performance of a Custom CNN and MobileNetV2 Transfer Learning model.

## Model Performance

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Test Accuracy | 81.71% | 80.89% |
| Test Loss | 0.4954 | 0.5010 |
| Macro F1 | 81.05% | 80.45% |
| Weighted F1 | 81.48% | 80.56% |

## Best Model

**Custom CNN** achieved the higher test accuracy and is selected as the better-performing model for this dataset.

## Analysis

- Custom CNN achieved slightly higher overall test accuracy.
- MobileNetV2 used significantly fewer trainable parameters.
- MobileNetV2 performed particularly well on the no_tumor class.
- Custom CNN performed particularly well on the pituitary class.
- Meningioma was a challenging class for both models.

## Conclusion

Based on test accuracy and overall F1 performance, the Custom CNN performed slightly better on this dataset. MobileNetV2 provides a more lightweight alternative with substantially fewer parameters.

The selected model should be treated as an educational and research model and not as a standalone medical diagnostic system.

## Comparison Chart

Saved at: `results/plots/model_accuracy_comparison.png`
