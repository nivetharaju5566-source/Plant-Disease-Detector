"""
app.py - Streamlit Web Application for Plant Disease Detection
RISE Internship - Project 8

Run with: streamlit run app.py
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .title-text {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #2e7d32, #66bb6a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #555; font-size: 1rem; margin-bottom: 2rem; }
    .result-healthy {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 6px solid #2e7d32; border-radius: 12px;
        padding: 1.5rem; margin: 1rem 0;
    }
    .result-diseased {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border-left: 6px solid #e65100; border-radius: 12px;
        padding: 1.5rem; margin: 1rem 0;
    }
    .metric-card {
        background: white; border-radius: 10px;
        padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center; margin: 0.5rem 0;
    }
    .info-box {
        background: #f1f8e9; border-radius: 8px;
        padding: 1rem; border: 1px solid #c5e1a5;
    }
    .stProgress .st-bo { background-color: #4caf50; }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached(model_path):
    """Load model with caching."""
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess_image(img, img_size=(128, 128)):
    """Preprocess image for model inference."""
    img = img.convert('RGB').resize(img_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0), img


def predict(model, img_array, threshold=0.5):
    """Run inference and return prediction details."""
    raw_prob = float(model.predict(img_array, verbose=0)[0][0])

    # Class mapping depends on how ImageDataGenerator sorted folders
    # 'diseased' = class 0 (index 0), 'healthy' = class 1 (index 1)
    # So: sigmoid > threshold → class 1 → healthy
    is_healthy = raw_prob > threshold
    confidence = raw_prob if is_healthy else (1 - raw_prob)

    label = "Healthy" if is_healthy else "Diseased"
    color = "green" if is_healthy else "red"
    emoji = "🌱" if is_healthy else "🍂"

    return {
        'label': label,
        'is_healthy': is_healthy,
        'confidence': confidence,
        'raw_prob': raw_prob,
        'color': color,
        'emoji': emoji
    }


def plot_confidence_gauge(confidence, label, is_healthy):
    """Create a Plotly gauge chart for confidence."""
    color = "#2e7d32" if is_healthy else "#e65100"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        number={'suffix': "%", 'font': {'size': 36, 'color': color}},
        title={'text': f"Confidence: {label}", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffecb3'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#dcedc8'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_probability_bar(raw_prob):
    """Horizontal bar showing class probabilities."""
    healthy_p = raw_prob * 100
    diseased_p = (1 - raw_prob) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Diseased', x=[diseased_p], y=['Probability'],
        orientation='h', marker_color='#ef6c00',
        text=f'{diseased_p:.1f}%', textposition='inside'
    ))
    fig.add_trace(go.Bar(
        name='Healthy', x=[healthy_p], y=['Probability'],
        orientation='h', marker_color='#388e3c',
        text=f'{healthy_p:.1f}%', textposition='inside'
    ))
    fig.update_layout(
        barmode='stack', height=100,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True, paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation='h', x=0.3, y=1.3)
    )
    return fig


def load_training_report():
    """Load training metrics from JSON report if available."""
    try:
        with open('outputs/training_report.json') as f:
            return json.load(f)
    except Exception:
        return None


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/000000/seedling.png", width=80)
    st.markdown("## ⚙️ Settings")

    model_path = st.text_input("Model Path", value="models/plant_disease_model.h5")
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Adjust classification boundary (default: 0.5)")

    st.divider()
    st.markdown("### 📌 About")
    st.markdown("""
    **RISE Internship - Project 8**  
    Plant Disease Detection using CNN  
    
    - 🏗️ Model: Custom CNN / MobileNetV2  
    - 📐 Input: 128×128 RGB images  
    - 🎯 Task: Binary Classification  
    - ⚙️ Framework: TensorFlow + Keras  
    """)

    report = load_training_report()
    if report:
        st.divider()
        st.markdown("### 📊 Model Performance")
        st.metric("Best Val Accuracy",
                  f"{report.get('best_val_accuracy', 0):.2%}")
        st.metric("Final Train Accuracy",
                  f"{report.get('final_train_accuracy', 0):.2%}")
        st.metric("Model Type", report.get('model_type', 'custom').upper())


# ─── Main App ────────────────────────────────────────────────────────────────
st.markdown('<div class="title-text">🌿 Plant Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered visual disease detector for agriculture & rural tech awareness</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Detect Disease", "📊 Model Info", "📖 How to Use"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Detection
# ═══════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("### 📤 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a clear photo of a plant leaf"
        )

        # Camera input option
        use_camera = st.checkbox("📷 Use Camera Instead")
        camera_image = None
        if use_camera:
            camera_image = st.camera_input("Take a photo of a leaf")

        # Determine source image
        source = camera_image or uploaded_file

        if source:
            img = Image.open(source)
            st.image(img, caption="Uploaded Leaf Image",
                     use_column_width=True)
            st.caption(f"Original size: {img.size[0]}×{img.size[1]} px")

    with col2:
        st.markdown("### 🤖 Analysis Results")

        if source is None:
            st.markdown("""
            <div class="info-box">
            <b>👈 Upload or capture a leaf image to get started</b><br><br>
            The AI will analyze the image and tell you whether the plant is:<br>
            🌱 <b>Healthy</b> — No visible disease detected<br>
            🍂 <b>Diseased</b> — Disease patterns found
            </div>
            """, unsafe_allow_html=True)
        else:
            # Load model
            model, err = load_model_cached(model_path)

            if err or model is None:
                st.error(f"❌ Could not load model from `{model_path}`")
                st.info("👉 Run `python train.py` first to train and save the model.")
            else:
                with st.spinner("🔬 Analyzing leaf..."):
                    img_array, processed_img = preprocess_image(img)
                    result = predict(model, img_array, threshold=threshold)

                # Result banner
                css_class = "result-healthy" if result['is_healthy'] else "result-diseased"
                st.markdown(f"""
                <div class="{css_class}">
                    <h2 style="margin:0">{result['emoji']} {result['label']}</h2>
                    <p style="margin:0.3rem 0 0 0; color:#333;">
                        Confidence: <b>{result['confidence']:.1%}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Recommendation
                if result['is_healthy']:
                    st.success("✅ This leaf appears **healthy**. No disease treatment needed.")
                else:
                    st.error("⚠️ Disease detected. Consult an agronomist for treatment advice.")
                    st.markdown("""
                    **Common treatments:**
                    - Apply appropriate fungicide/pesticide
                    - Remove and dispose infected leaves
                    - Improve air circulation around plants
                    - Check soil pH and nutrients
                    """)

                # Confidence Gauge
                st.plotly_chart(
                    plot_confidence_gauge(result['confidence'], result['label'], result['is_healthy']),
                    use_container_width=True
                )

                # Probability breakdown
                st.markdown("**Class Probability Breakdown:**")
                st.plotly_chart(plot_probability_bar(result['raw_prob']),
                                use_container_width=True)

                # Raw score
                with st.expander("🔢 Raw Model Output"):
                    st.code(f"""
Model sigmoid output : {result['raw_prob']:.6f}
Decision threshold   : {threshold}
Predicted class      : {result['label']}
Confidence           : {result['confidence']:.4f} ({result['confidence']:.2%})
""")

# ═══════════════════════════════════════════════════════════
# TAB 2 — Model Info
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏗️ Model Architecture")
    model_loaded, err = load_model_cached(model_path)

    if model_loaded:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Parameters", f"{model_loaded.count_params():,}")
        col_b.metric("Model Size (approx.)", f"{model_loaded.count_params() * 4 / 1e6:.2f} MB")
        col_c.metric("Input Shape", "128 × 128 × 3")

        # Layer table
        layers_info = []
        for layer in model_loaded.layers:
            try:
                params = layer.count_params()
            except Exception:
                params = 0
            layers_info.append({
                'Layer': layer.name,
                'Type': type(layer).__name__,
                'Output Shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                'Parameters': f"{params:,}"
            })

        import pandas as pd
        df = pd.DataFrame(layers_info)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ Load a trained model to see architecture details.")

    st.divider()
    st.markdown("### 📊 Training Metrics")

    report = load_training_report()
    if report:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Val Accuracy", f"{report.get('best_val_accuracy', 0):.2%}")
        col2.metric("Final Train Accuracy", f"{report.get('final_train_accuracy', 0):.2%}")
        col3.metric("Epochs Trained", report.get('epochs_trained', 'N/A'))
        col4.metric("Architecture", report.get('model_type', 'custom').upper())

        if os.path.exists('outputs/training_history.png'):
            st.image('outputs/training_history.png', caption="Training Curves",
                     use_column_width=True)
        if os.path.exists('outputs/confusion_matrix.png'):
            st.image('outputs/confusion_matrix.png', caption="Confusion Matrix",
                     use_column_width=True)
    else:
        st.info("📌 Train the model first to see performance charts here.")

# ═══════════════════════════════════════════════════════════
# TAB 3 — How to Use
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📖 Getting Started Guide")

    st.markdown("""
    ### Step 1 — Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

    ### Step 2 — Prepare Your Dataset
    ```
    plant_disease_detection/
    ├── data/
    │   ├── healthy/      ← add healthy leaf images (.jpg, .png)
    │   └── diseased/     ← add diseased leaf images (.jpg, .png)
    ```
    
    > **Free Dataset**: Download from [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
    > (sort images into the two class folders)

    ### Step 3 — Train the Model
    ```bash
    # Default (Custom CNN, 20 epochs)
    python train.py

    # Use MobileNetV2 transfer learning
    python train.py --model mobilenet

    # Custom parameters
    python train.py --epochs 30 --batch_size 64 --lr 0.0005

    # With fine-tuning (MobileNet only)
    python train.py --model mobilenet --fine_tune
    ```

    ### Step 4 — Launch the App
    ```bash
    streamlit run app.py
    ```
    Open your browser at **http://localhost:8501**

    ---

    ### 📁 Project Structure
    ```
    plant_disease_detection/
    ├── app.py               ← Streamlit web app (this file)
    ├── model.py             ← CNN architecture definitions
    ├── train.py             ← Training pipeline
    ├── requirements.txt     ← Python dependencies
    ├── data/
    │   ├── healthy/
    │   └── diseased/
    ├── models/
    │   └── plant_disease_model.h5  ← Saved after training
    ├── outputs/
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   └── training_report.json
    └── utils/
        └── data_utils.py    ← Data processing utilities
    ```

    ---
    ### 🎯 Tips for Best Results
    - Use **clear, well-lit** images of individual leaves
    - Collect **at least 200+ images** per class for good accuracy
    - Use **data augmentation** (enabled by default) when dataset is small
    - Try **MobileNet transfer learning** for better accuracy with less data
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>🌿 Plant Disease Detection — RISE Internship Project 8 | "
    "Built with TensorFlow + Streamlit</small></center>",
    unsafe_allow_html=True
)
