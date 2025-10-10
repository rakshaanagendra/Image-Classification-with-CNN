import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from src.class_names import class_names
import tensorflow as tf

# 1️⃣ Page setup
st.set_page_config(page_title="Pet Classifier", page_icon="🐶", layout="centered")
st.title("🐾 Pet Image Classifier")
st.write("Upload a pet image, and the model will predict its breed or category!")

# 2️⃣ Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("saved_models/pet_classifier.keras")
    return model

model = load_trained_model()

# 3️⃣ Image upload section
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 4️⃣ Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # 5️⃣ Preprocess the image
    img = tf.convert_to_tensor(np.array(img))
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    img_array = np.expand_dims(img, axis=0)

    # 6️⃣ Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_names[predicted_class]

    # 7️⃣ Show result
    st.subheader("🧠 Prediction Result:")
    st.write(f"Predicted: **{predicted_label}**")
    confidence = np.max(prediction) * 100
    st.write(f"Confidence: **{confidence:.2f}%**")
    st.success("✅ Prediction complete!")
else:
    st.info("👆 Please upload an image to start prediction.")
