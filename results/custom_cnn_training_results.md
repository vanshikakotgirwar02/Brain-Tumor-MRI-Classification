# Custom CNN Training Results

## Project

**Brain Tumor MRI Image Classification**

## Dataset

  Dataset        Images
  ------------ --------
  Train           1,695
  Validation        502
  Test              246

### Classes

-   Glioma
-   Meningioma
-   No Tumor
-   Pituitary

## Model

A custom Convolutional Neural Network (CNN) was trained for 4-class
brain MRI image classification.

### Input

-   Image size: `224 × 224 × 3`
-   Batch size: `32`
-   Optimizer: Adam
-   Learning rate: `0.0001`
-   Maximum epochs: `20`
-   Early stopping patience: `5`
-   Dropout: `0.5`

### Architecture

1.  Data augmentation
2.  Rescaling
3.  Conv2D --- 32 filters
4.  MaxPooling2D
5.  Conv2D --- 64 filters
6.  MaxPooling2D
7.  Conv2D --- 128 filters
8.  MaxPooling2D
9.  Conv2D --- 256 filters
10. MaxPooling2D
11. Flatten
12. Dense --- 256 units
13. Dropout --- 0.5
14. Dense --- 4 units with softmax activation

**Total parameters:** 13,234,756

## Training Results

The model achieved its best validation accuracy at **Epoch 12**.

  Metric                                Best observed result
  ----------------------------------- ----------------------
  Best validation accuracy                        **84.46%**
  Best validation loss                            **0.4824**
  Epoch of best validation accuracy                   **12**
  Training accuracy at Epoch 12                   **79.82%**
  Training loss at Epoch 12                       **0.5518**

### Selected Epoch Results

    Epoch   Training Accuracy   Validation Accuracy   Validation Loss
  ------- ------------------- --------------------- -----------------
        1              51.68%                59.96%            0.9699
        2              61.42%                66.93%            0.8060
        3              66.19%                72.11%            0.7108
        4              71.98%                75.50%            0.6736
        5              74.75%                77.29%            0.5944
        7              75.87%                78.49%            0.6341
        8              78.82%                81.47%            0.5507
       12              79.82%            **84.46%**        **0.4824**
       19              84.48%                78.09%            0.6026

## Saved Files

### Best model

``` text
models/custom_cnn.keras
```

This is the checkpoint with the best validation accuracy.

### Final model

``` text
models/custom_cnn_final.keras
```

### Training plots

``` text
results/plots/custom_cnn_accuracy.png
results/plots/custom_cnn_loss.png
```

## Observation

The model's validation accuracy improved from **59.96% in Epoch 1** to
**84.46% in Epoch 12**.

After Epoch 12, validation accuracy became less stable while training
accuracy continued to increase. This indicates that the best checkpoint
should be used for subsequent evaluation rather than relying on the
final training epoch.

## Next Step

The next stage is to evaluate the saved best CNN model on the **unseen
test dataset** using:

-   Test accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix
-   Per-class performance
