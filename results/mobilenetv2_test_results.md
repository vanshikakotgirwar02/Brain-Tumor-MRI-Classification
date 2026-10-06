# MobileNetV2 Test Results

## Dataset

- Test images: 246
- Number of classes: 4
- Image size: 224 × 224
- Classes: glioma, meningioma, no_tumor, pituitary

## Model

- Architecture: MobileNetV2
- Transfer Learning: ImageNet
- Pretrained layers: Frozen

## Test Performance

- **Test Accuracy:** 80.89%
- **Test Loss:** 0.5010

## Classification Report

```text
              precision    recall  f1-score   support

      glioma     0.8642    0.8750    0.8696        80
  meningioma     0.8043    0.5873    0.6789        63
    no_tumor     0.9744    0.7755    0.8636        49
   pituitary     0.6750    1.0000    0.8060        54

    accuracy                         0.8089       246
   macro avg     0.8295    0.8095    0.8045       246
weighted avg     0.8293    0.8089    0.8056       246

```

## Confusion Matrix

The confusion matrix is saved at `results/plots/mobilenetv2_confusion_matrix.png`.

## Notes

This model is intended for educational and research purposes and should not be used as a standalone medical diagnostic system.
