# 🧠 NeuroVision AI — Brain Tumor MRI Image Classification

An AI-powered deep learning project for classifying brain MRI images into four categories using a Custom Convolutional Neural Network (CNN) and MobileNetV2 transfer learning.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not replace evaluation by a qualified medical professional.

---

## 📌 Project Overview

NeuroVision AI uses deep learning techniques to analyze brain MRI images and classify them into:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The project compares two approaches:

1. **Custom CNN**
2. **MobileNetV2 Transfer Learning**

After evaluation on an unseen test dataset, the Custom CNN achieved the highest overall test accuracy.

---

## 🎯 Objectives

- Inspect and validate the MRI dataset
- Analyze image quality and class distribution
- Preprocess and augment MRI images
- Build a Custom CNN classification model
- Apply MobileNetV2 transfer learning
- Evaluate both models using multiple metrics
- Compare model performance
- Build an interactive Streamlit application

---

## 📊 Dataset

The dataset contains **2,443 MRI images** divided into training, validation and testing sets.

| Dataset | Images |
|---|---:|
| Training | 1,695 |
| Validation | 502 |
| Testing | 246 |
| **Total** | **2,443** |

### Classes

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Glioma | 564 | 161 | 80 |
| Meningioma | 358 | 124 | 63 |
| No Tumor | 335 | 99 | 49 |
| Pituitary | 438 | 118 | 54 |

All images were successfully validated during dataset inspection, with **0 corrupted images**.

---

## 🔬 Project Workflow

```text
Dataset
   ↓
Dataset Inspection
   ↓
Image Quality Analysis
   ↓
Dataset Visualization
   ↓
Class Distribution
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
┌───────────────────────┐
│                       │
│   Custom CNN          │
│                       │
└───────────────────────┘
           │
           ↓
      Model Evaluation

           +

┌───────────────────────┐
│                       │
│   MobileNetV2         │
│   Transfer Learning   │
│                       │
└───────────────────────┘
           │
           ↓
      Model Evaluation
           │
           ↓
    Model Comparison
           │
           ↓
     Streamlit App

     