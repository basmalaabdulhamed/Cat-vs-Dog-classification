# 🐱 Cat vs Dog Image Classification

> Binary image classification using CNN from Scratch and Transfer Learning (MobileNetV2)

**Elsewedy University of Technology — Image Processing Course | Spring 2026**

---

## 👥 Team Members

| Name | ID |
|------|----|
| Basmala Mohamed | 240103118 |
| Sama Walid | 240102884 |
| Mahmoud Hesham | 240101375 |

**Supervisor:** Eng. Hend Adel Ahmed

---

## 📌 Project Overview

This project builds and compares two deep learning models to classify images as either a **Cat** or a **Dog**:

- **Model 1:** Convolutional Neural Network (CNN) built from scratch
- **Model 2:** Transfer Learning using MobileNetV2 (pretrained on ImageNet)

A Streamlit web application is included for real-time prediction.

---

## 📊 Results

| Metric | CNN from Scratch | Transfer Learning |
|--------|:-:|:-:|
| Accuracy | 74.22% | **97.78%** |
| Precision | 72.06% | **96.94%** |
| Recall | 79.11% | **98.67%** |
| F1-Score | 75.42% | **97.80%** |
| AUC | 83.20% | **99.91%** |

---

## 📁 Project Structure

```
Cat-vs-Dog-classification/
│
├── notebooks/
│   ├── analysis.ipynb             # Phase 1 — dataset exploration & histogram analysis
│   ├── preprocessing.ipynb        # Resizing, normalization, augmentation, splitting
│   ├── cnn_model.ipynb            # CNN from scratch — training & evaluation
│   └── transfer_learning.ipynb    # MobileNetV2 transfer learning & fine-tuning
│
├── results/
│   ├── cnn_training_curves.png
│   ├── cnn_confusion_matrix.png
│   ├── cnn_roc_curve.png
│   ├── tl_training_curves.png
│   ├── tl_confusion_matrix.png
│   └── tl_roc_curve.png
│
├── app.py                         # Streamlit web application
├── requirements.txt
└── .gitignore
```

> **Note:** The `data/` and `models/` folders are excluded from this repo due to file size limits.
> Download the dataset from [Kaggle](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset).

---

## 🗃️ Dataset

- **Source:** [Kaggle — Dogs vs. Cats](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)
- **Subset used:** 6,000 images (3,000 cats + 3,000 dogs)
- **Split:** 70% train / 15% validation / 15% test
- **Class balance:** Perfectly balanced (50% each)

---

## 🧠 Model Architectures

### CNN from Scratch
4 convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPooling) with increasing filters (32 → 64 → 128 → 256), followed by Flatten → Dense(512) → Dropout(0.5) → Sigmoid output.

### Transfer Learning — MobileNetV2
- **Phase 1:** MobileNetV2 base frozen, custom head trained (GlobalAvgPool → Dense(256) → Dropout → Dense(128) → Sigmoid)
- **Phase 2:** Last 30 layers unfrozen for fine-tuning at learning rate 0.0001

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/basmalaabdulhamed/Cat-vs-Dog-classification.git
cd Cat-vs-Dog-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

> Make sure you have the trained model file `models/tl_final.keras` in the `models/` folder before running the app.

---

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, PIL (Pillow)
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

---

## 📈 Visualizations

All result plots are saved in the `results/` folder:

- Training & Validation accuracy/loss curves
- Confusion matrices (heatmap)
- ROC curves with AUC scores
- Sample prediction grids
