# MobileNetV2 Transfer Learning Results

## Dataset

- Training images: 1,695
- Validation images: 502
- Number of classes: 4
- Image size: 224 × 224

## Model

- Architecture: MobileNetV2
- Transfer learning: ImageNet pretrained weights
- Pretrained layers: Frozen
- Optimizer: Adam
- Learning rate: 0.0001

## Results

- **Best Validation Accuracy:** 80.48%
- **Best Accuracy Epoch:** 12
- **Lowest Validation Loss:** 0.5132
- **Lowest Loss Epoch:** 12

## Saved Files

- `models/mobilenetv2_transfer.keras`
- `models/mobilenetv2_transfer_final.keras`
- `results/plots/mobilenetv2_accuracy.png`
- `results/plots/mobilenetv2_loss.png`
