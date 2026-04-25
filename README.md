#  Pneumonia Detection using ResNet18 (PyTorch)

This project focuses on detecting **pneumonia from chest X-ray images** using a deep learning model based on **ResNet18 with transfer learning**.

---

##  Key Highlights

*  Merged original train + validation datasets
*  Applied **Stratified 80/20 split** for better validation
*  Used **ResNet18 (ImageNet pretrained)**
*  Fine-tuning: Unfreezes **layer4 + FC head** only
*  Handled class imbalance using **weighted loss (pos_weight)**
*  Visualized training (accuracy & loss curves)
* 📊 Evaluated using:

  * Confusion Matrix
  * Classification Report

---

##  Dataset

Chest X-ray dataset with 2 classes:

* NORMAL
* PNEUMONIA



---

##  Data Strategy (Important Fix)

The original validation dataset was too small, leading to unreliable validation performance.

✔ Solution:

* Combined **train + validation datasets**
* Applied **StratifiedShuffleSplit**
* New split:

  * 80% Training
  * 20% Validation

This improved the reliability of validation accuracy.

---

## Model Architecture

* Backbone: **ResNet18**
* Pretrained on ImageNet
* Custom classifier:

  * Linear (512 → 256)
  * ReLU
  * Dropout (0.5)
  * Output layer (Binary classification)

Loss Function:

* `BCEWithLogitsLoss` with class weighting

---

## 🧪 Training Configuration

| Parameter      | Value    |
| -------------- | -------- |
| Batch Size     | 32       |
| Epochs         | 20       |
| Initial LR     | 1e-4     |
| Fine-tune LR   | 1e-5     |
| Weight Decay   | 2e-4     |
| Early Stopping | 5 epochs |

---

## 📊 Results

| Metric              | Value      |
| ------------------- | ---------- |
| Validation Accuracy | **97.99%** |
| Test Accuracy       | **83.81%** |

###  Classification Report (Test Set)

| Class     | Precision | Recall   | F1-score |
| --------- | --------- | -------- | -------- |
| NORMAL    | 0.98      | 0.58     | 0.73     |
| PNEUMONIA | 0.80      | **0.99** | 0.88     |

---

## 🩺 Key Insight (Very Important)

In medical diagnosis tasks, **recall is more important than precision**, especially for the disease class.

* Missing a pneumonia case (**false negative**) can be dangerous
* False positives (healthy predicted as pneumonia) are less harmful

✔ This model achieves:

👉 **Recall = 0.99 for Pneumonia**
→ Almost all infected patients are correctly detected

⚠️ Trade-off:

* Lower recall for NORMAL class (0.58)
  → Model is slightly biased toward predicting pneumonia (safer side)

---

## 📈 Visualizations

* Accuracy vs Epoch
* Loss vs Epoch
* Confusion Matrix

Saved outputs:

* `training_history_resnet18.png`
* `confusion_matrix_resnet18.png`

---

## How to Run (Google Colab)

1. Upload dataset ZIP to Google Drive
2. Update path in code:

```python id="m7k2z9"
ZIP_PATH =  "/content/drive/MyDrive/archive (1).zip"
```

3. Run:

```bash id="r4k8d1"
python dl_project_resnet18.py
```

---

## 📦 Requirements

```id="t8p3l2"
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
```

---

## ⚠️ Limitations

* Test accuracy lower than validation → indicates slight overfitting
* Model tends to over-predict pneumonia cases
* Dataset size and distribution may affect generalization

---

## 🔮 Future Improvements

* Try **ResNet50 / EfficientNet**
* Threshold tuning (precision-recall balance)
* K-fold cross validation
* Grad-CAM for model interpretability
* Hyperparameter tuning

---

## 👩‍💻 Author

Kanika Dhamija
B.E. Computer Engineering (2023–2027)

---
