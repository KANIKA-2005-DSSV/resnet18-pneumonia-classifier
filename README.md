# 🫁 Pneumonia Detection from Chest X-Rays using Deep Learning

> Automated binary classification of chest X-ray images (NORMAL vs PNEUMONIA) using four deep learning models — Custom CNN, MobileNetV2, ResNet18, and DenseNet121.

---

## 📌 Overview

Pneumonia is a leading cause of death in children under five, accounting for around 740,000 deaths annually. Diagnosing it requires a trained radiologist to interpret chest X-rays — a resource that is often unavailable in low-income or remote healthcare settings.

This project builds and compares four deep learning models that can automatically classify a chest X-ray as **NORMAL** or **PNEUMONIA**, with the goal of supporting faster and more accessible screening. The best model (DenseNet121) achieves **90.87% accuracy** and **0.959 AUC** on the held-out test set.

---

## 📂 Dataset

**Source:** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
**Original paper:** Kermany et al., *Cell*, 2018

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train (original) | 1,341 | 3,875 | 5,216 |
| Val (original) | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

> **Note:** The original validation set (16 images) is too small for reliable training signals. ResNet18 and DenseNet121 merge train + val and re-split at 80/20 stratified (~1,047 validation images), while Custom CNN and MobileNetV2 use the original splits.

- **Image type:** Pediatric anterior-posterior chest X-rays (JPEG, grayscale)
- **Class distribution:** ~74% PNEUMONIA, ~26% NORMAL — class imbalance handled via weighted loss in all models

---

## 🧠 Models Used

### 1. Custom CNN (Baseline)
A CNN built from scratch using Keras with no pretrained weights. Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement before training.

**Architecture:**  
`Conv2D(32) → Conv2D(64) → Conv2D(128) → GlobalAvgPool → Dense(128) → Dense(1, Sigmoid)`

### 2. MobileNetV2
Transfer learning using MobileNetV2 pretrained on ImageNet. Uses depthwise separable convolutions for high efficiency. Custom classification head with two dense layers.

**Training:** Two-phase — head warm-up (LR=1e-4) → partial fine-tuning (LR=1e-5).

### 3. ResNet18
Transfer learning using ResNet18 pretrained on ImageNet, implemented in **PyTorch**. Uses residual (skip) connections to solve the vanishing gradient problem.

**Training:** Two-phase — frozen backbone for epochs 1–5 (LR=1e-4), then full fine-tuning from epoch 6 (LR=1e-5). Uses `BCEWithLogitsLoss` with `pos_weight` for class imbalance and `AdamW` with `CosineAnnealingLR`.

### 4. DenseNet121 ⭐ (Best Model)
Transfer learning using DenseNet121 — the same architecture behind Stanford's CheXNet. Dense connectivity: each layer receives feature maps from all previous layers in its block, maximising feature reuse.

**Training:** Two-phase with mixed-precision (float16) training on T4 GPU. Top 120/427 layers unfrozen in Phase 2. XLA JIT compilation for ~12% GPU throughput gain.

---

## ⚙️ Methodology

### Data Preprocessing

| Step | Training | Validation / Test |
|------|----------|-------------------|
| Resize | 224 × 224 | 224 × 224 |
| Augmentation | Flip, rotation, zoom, shift, brightness | None |
| Normalization | ImageNet stats / model-specific `preprocess_input` | Same |
| CLAHE | Custom CNN only | — |

### Transfer Learning Strategy

All three transfer learning models follow the same two-phase approach:

**Phase 1 — Feature Extraction:** Backbone layers are frozen. Only the custom classification head is trained at a higher learning rate. This lets the head learn to use pretrained features without disturbing them.

**Phase 2 — Fine-Tuning:** Deeper backbone layers are unfrozen and trained at a 10× lower learning rate, adapting the model's feature representations to chest X-ray data while avoiding catastrophic forgetting of ImageNet features.

### Loss Function & Optimizer

| Model | Framework | Loss | Optimizer | Class Imbalance Handling |
|-------|-----------|------|-----------|--------------------------|
| Custom CNN | TensorFlow/Keras | Binary Cross-Entropy | Adam (1e-3) | `class_weight` {NORMAL: 1.9448, PNEUMONIA: 0.6730} |
| MobileNetV2 | TensorFlow/Keras | Binary Cross-Entropy | Adam (1e-4 / 1e-5) | `class_weight` {NORMAL: 1.9448, PNEUMONIA: 0.6730} |
| ResNet18 | PyTorch | BCEWithLogitsLoss | AdamW (wd=2e-4) | `pos_weight=0.35` (NORMAL/PNEUMONIA ratio) |
| DenseNet121 | TensorFlow/Keras | Binary Cross-Entropy | Adam (1e-4 / 1e-5) | `class_weight` {NORMAL: 1.9392, PNEUMONIA: 0.6737} |

---

## 📊 Evaluation Metrics

Each model is evaluated on the held-out test set (624 images, never seen during training) using:

- **Accuracy** — overall correct predictions
- **AUC (ROC)** — ability to discriminate between classes across all thresholds
- **Sensitivity / Recall** — proportion of actual pneumonia cases correctly identified (most critical in medical screening)
- **Specificity** — proportion of actual normal cases correctly identified
- **Precision** — of all predicted pneumonia, how many were correct
- **F1-Score (weighted)** — harmonic mean of precision and recall
- **Confusion Matrix** — TP, TN, FP, FN breakdown

---

## 📈 Results & Comparison

### Overall Performance

| Model | Accuracy | AUC | Sensitivity | Specificity | Precision | F1 (wt) | Test Loss |
|-------|----------|-----|-------------|-------------|-----------|----------|-----------|
| Custom CNN (baseline) | 73.08% | 0.791 | 77.95% | 64.96% | 0.79 | 0.73 | 0.541 |
| MobileNetV2 | 89.74% | 0.974 | 97.95% | 76.07% | 0.87 | 0.89 | — |
| ResNet18 | 83.81% | 0.870 | 99.23% | 58.12% | 0.80 | 0.83 | 0.618 |
| **DenseNet121 ⭐** | **90.87%** | **0.959** | **96.15%** | **82.05%** | **0.90** | **0.91** | **0.306** |

### Confusion Matrix Breakdown

| Model | TP (correct pneumonia) | TN (correct normal) | FP (false alarm) | FN (missed pneumonia) | Total Errors |
|-------|------------------------|----------------------|-------------------|-----------------------|--------------|
| Custom CNN | 304 | 152 | 82 | 86 | 168 |
| MobileNetV2 | 382 | 178 | 56 | 8 | 64 |
| ResNet18 | 387 | 136 | 98 | 3 | 101 |
| **DenseNet121** | **375** | **192** | **42** | **15** | **57** |

### Per-Class Classification Report — DenseNet121 (Best Model)

```
              precision    recall  f1-score   support

      NORMAL       0.93      0.82      0.87       234
   PNEUMONIA       0.90      0.96      0.93       390

    accuracy                           0.91       624
   macro avg       0.91      0.89      0.90       624
weighted avg       0.91      0.91      0.91       624
```

### Per-Class Classification Report — ResNet18

```
              precision    recall  f1-score   support

      NORMAL       0.98      0.58      0.73       234
   PNEUMONIA       0.80      0.99      0.88       390

    accuracy                           0.84       624
   macro avg       0.89      0.79      0.81       624
weighted avg       0.87      0.84      0.83       624
```

---

## 📉 Visualizations

Each model notebook includes:

- **Training & Validation Accuracy/Loss Curves** — plotted across all epochs; a vertical line marks the phase transition in two-phase models
- **Confusion Matrix Heatmap** — Seaborn heatmap showing TP/TN/FP/FN counts
- **ROC Curve** — plots True Positive Rate vs False Positive Rate; shaded AUC area shown
- **Class Distribution Bar Chart** — visualizes the NORMAL/PNEUMONIA imbalance in the training set

---

## 💡 Key Insights

**What worked well:**

- Transfer learning decisively outperformed training from scratch — all three transfer models beat the custom CNN by 10–18 percentage points. Pre-trained ImageNet features transfer well to chest X-ray classification.
- DenseNet121's dense connectivity provides the richest feature representations for subtle radiological patterns, explaining its best-in-class specificity and F1.
- Proper 20% validation splits (ResNet18, DenseNet121) enabled effective fine-tuning and reliable early stopping — producing significantly better test results than the 16-image original val set.
- AdamW with CosineAnnealingLR gave ResNet18 the smoothest convergence and highest recall (99.23%, FN=3).

**What didn't work / limitations:**

- The original 16-image validation set crippled MobileNetV2's Phase 2 fine-tuning — EarlyStopping triggered after just 1 epoch, preventing any fine-tuning gain.
- ResNet18 overfit significantly: val_accuracy peaked at 97.99% during training but test accuracy was only 83.81% — a 14.2 percentage point gap caused by the full fine-tuning phase after epoch 6. Earlier EarlyStopping would have helped.
- ResNet18's near-perfect recall (FN=3) comes with 98 false positives — a 42% false positive rate on normal patients that would be clinically impractical.
- No model includes visual explanations (Grad-CAM). In medical AI, explainability is critical for clinician trust.

---

## ✅ Conclusion

This project demonstrates that deep learning, specifically transfer learning with pretrained convolutional networks, can effectively detect pneumonia from chest X-rays with clinically meaningful accuracy. DenseNet121 is the clear best model, offering the best overall balance of accuracy (90.87%), specificity (82.05%), F1 (0.91), and lowest test loss (0.306). For scenarios where minimising missed diagnoses is the sole priority, ResNet18 (FN=3) or MobileNetV2 (FN=8) may be preferred, at the cost of significantly more false alarms.

---

## 🚀 Future Improvements

- **Grad-CAM visualisation** — highlight which regions of the X-ray influenced the model's prediction, enabling clinician review and trust
- **Multi-class extension** — differentiate bacterial vs viral pneumonia, or detect other conditions (COVID-19, TB)
- **External validation** — test on adult chest X-ray datasets to assess generalisability beyond pediatric cases
- **Ensemble model** — combine predictions from all four models for potentially higher and more robust performance
- **Vision Transformer (ViT) / ConvNeXt V2** — explore 2024 state-of-the-art architectures that have shown AUC > 0.99 on chest X-ray tasks
- **Mobile deployment** — convert the best model to ONNX or TensorFlow Lite for deployment in resource-constrained clinical settings

---

## 🛠️ Tech Stack

| Component | Tools |
|-----------|-------|
| Deep Learning (CNN, MobileNetV2, DenseNet121) | TensorFlow 2.x / Keras |
| Deep Learning (ResNet18) | PyTorch |
| Data handling | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Metrics | scikit-learn |
| Training environment | Google Colab (Tesla T4 GPU) |
| Mixed precision | TensorFlow `mixed_float16` policy |

---

## 📁 Repository Structure

```
pneumonia-detection/
│
├── CNN.ipynb                          # Custom CNN baseline
├── MobileNetV2.ipynb                  # MobileNetV2 transfer learning
├── Resnet18_pneumonia_classifier.ipynb # ResNet18 (PyTorch)
├── DenseNet121.ipynb                  # DenseNet121 (best model)
│
├── README.md


## 👩‍💻 Author

Kanika Dhamija
B.E. Computer Engineering (2023–2027)

---
