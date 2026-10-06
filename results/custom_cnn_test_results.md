# Custom CNN Test Results

## Dataset

- Test images: 246
- Number of classes: 4
- Image size: 224 × 224
- Classes: glioma, meningioma, no_tumor, pituitary

## Model

- Model: Custom CNN
- Model file: `models/custom_cnn.keras`

## Test Performance

- **Test Accuracy:** 81.71%
- **Test Loss:** 0.4954

## Classification Report

```text
              precision    recall  f1-score   support

      glioma     0.9315    0.8500    0.8889        80
  meningioma     0.7500    0.6190    0.6783        63
    no_tumor     0.7000    0.8571    0.7706        49
   pituitary     0.8525    0.9630    0.9043        54

    accuracy                         0.8171       246
   macro avg     0.8085    0.8223    0.8105       246
weighted avg     0.8216    0.8171    0.8148       246

```

## Confusion Matrix

The confusion matrix is saved at `results/plots/custom_cnn_confusion_matrix.png`.

## Notes

This model is intended for educational and research purposes and should not be used as a standalone medical diagnostic system.
