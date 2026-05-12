from enum import auto

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# ── Load Model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / 'models' / 'tl_final.keras'
    return tf.keras.models.load_model(model_path)

model = load_model()

# ── UI ─────────────────────────────────────────────────────────────────
st.title("🐱🐶 Cat vs Dog Classifier")
st.markdown("**Upload an image and the model will predict whether it's a Cat or a Dog!**")
st.markdown("---")

# Model info
with st.expander("ℹ️ About this model"):
    st.markdown("""
    - **Model:** MobileNetV2 Transfer Learning
    - **Dataset:** 6,000 images (3,000 cats / 3,000 dogs)
    - **Test Accuracy:** 97.78%
    - **AUC Score:** 0.9991
    """)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=auto)

    # Preprocess
    img_resized  = image.resize((224, 224))
    img_array    = np.array(img_resized) / 255.0
    img_expanded = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner('Analyzing...'):
        prediction = model.predict(img_expanded, verbose=0)[0][0]

    # Results
    st.markdown("---")
    if prediction > 0.5:
        confidence = prediction * 100
        st.success(f"## 🐶 It's a DOG!")
        st.metric("Confidence", f"{confidence:.2f}%")
    else:
        confidence = (1 - prediction) * 100
        st.success(f"## 🐱 It's a CAT!")
        st.metric("Confidence", f"{confidence:.2f}%")

    # Confidence bar
    st.markdown("### Prediction Probability")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🐱 Cat", f"{(1-prediction)*100:.2f}%")
    with col2:
        st.metric("🐶 Dog", f"{prediction*100:.2f}%")

    st.progress(float(prediction))
    st.caption("0% = Cat → 100% = Dog")

else:
    st.info("👆 Please upload an image to get started!")
    st.markdown("### Sample images to try:")
    st.markdown("- A photo of your cat or dog 🐾")
    st.markdown("- Any cat or dog image from Google")