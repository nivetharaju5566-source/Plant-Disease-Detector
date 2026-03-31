# 🌿 Plant Disease Detection using CNN
**RISE Internship — Project 8 | Tamizhan Skills**

> AI-powered visual plant disease detector that aligns with agriculture and rural tech awareness goals.

---

## 📌 Problem Statement
Farmers need tools to identify plant diseases early via mobile photos. Early detection helps prevent crop loss and reduces unnecessary pesticide use.

## 🎯 Objective
Train a CNN model to classify plant leaf images as **healthy** or **diseased** with high accuracy.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a Dataset
**Option A — Real Dataset (Recommended):**
- Download from [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- Sort images into `data/healthy/` and `data/diseased/`

**Option B — Quick Test with Synthetic Data:**
```bash
python generate_sample_data.py --count 50
```

### 3. Train the Model
```bash
# Custom CNN (faster)
python train.py

# MobileNetV2 Transfer Learning (better accuracy)
python train.py --model mobilenet

# All options
python train.py --model mobilenet --epochs 30 --batch_size 32 --fine_tune
```

### 4. Launch Web App
```bash
streamlit run app.py
```
Open browser at: **http://localhost:8501**

---

## 📁 Project Structure
```
plant_disease_detection/
├── app.py                    ← Streamlit web application
├── model.py                  ← CNN & MobileNetV2 architectures
├── train.py                  ← Full training pipeline
├── evaluate.py               ← Standalone evaluation script
├── generate_sample_data.py   ← Synthetic data generator (for testing)
├── requirements.txt
│
├── data/
│   ├── healthy/              ← Healthy leaf images (.jpg, .png)
│   └── diseased/             ← Diseased leaf images (.jpg, .png)
│
├── models/
│   └── plant_disease_model.h5  ← Saved model (after training)
│
├── outputs/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── training_report.json
│
└── utils/
    └── data_utils.py         ← Data preprocessing & augmentation
```

---

## 🏗️ Model Architectures

### Custom CNN (Default)
- 3 convolutional blocks (32 → 64 → 128 filters)
- Batch Normalization + Dropout for regularization
- GlobalAveragePooling → Dense layers
- ~1.5M parameters

### MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet (1.4M images)
- Lightweight & mobile-friendly
- Fine-tuning support
- ~2.3M parameters

---

## 🔧 Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `custom` | `custom` or `mobilenet` |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--img_size` | `128` | Image size (square) |
| `--fine_tune` | off | Fine-tune MobileNet layers |
| `--no_augment` | off | Disable data augmentation |

---

## 📊 Evaluation

```bash
# Evaluate on full dataset
python evaluate.py

# Test single image
python evaluate.py --image path/to/leaf.jpg
```

**Metrics tracked:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🌐 Web App Features

| Feature | Description |
|---------|-------------|
| Image Upload | Upload JPG/PNG leaf photos |
| Camera Input | Use device camera directly |
| Confidence Gauge | Visual confidence indicator |
| Probability Chart | Class probability breakdown |
| Threshold Control | Adjustable decision boundary |
| Training Report | View model performance metrics |

---

## 🔬 Data Augmentation Techniques
- Rotation (±30°)
- Width/Height shift (20%)
- Shear transformation (15%)
- Zoom (20%)
- Horizontal flip
- Brightness variation (80%–120%)

---

## 📈 Expected Performance

| Dataset Size | Expected Accuracy |
|---|---|
| 100–300 images | 70–80% |
| 500–1000 images | 80–88% |
| 1000+ images | 88–95%+ |

> Using transfer learning (MobileNetV2) typically improves accuracy by 5–10%.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow 2.x + Keras |
| Transfer Learning | MobileNetV2 (ImageNet) |
| Web App | Streamlit |
| Data Processing | NumPy, Pillow, OpenCV |
| Visualization | Matplotlib, Seaborn, Plotly |
| Evaluation | Scikit-learn |

---

## 💡 Tips for Better Results
1. **More data = better accuracy** — aim for 500+ images per class
2. **Use transfer learning** when dataset is small
3. **Augmentation** compensates for limited data
4. **Fine-tuning** after initial training improves MobileNet accuracy
5. Use **real plant disease images** (not synthetic) for deployment

---

## 📞 Contact
**RISE Internship | Tamizhan Skills**  
📞 +91 6383418100  
🌐 www.tamizhanskills.com  
📧 contact@tamizhanskills.com
