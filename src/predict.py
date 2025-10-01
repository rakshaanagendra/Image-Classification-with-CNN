import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import json
import os

def load_image(path, img_size=(224,224)):
    """Load and preprocess a single image for prediction."""
    img = Image.open(path).convert("RGB")
    img = img.resize(img_size)
    arr = np.array(img).astype("float32") / 255.0
    return arr

def main(image_path):
    # ✅ load class map from correct path
    cm_path = os.path.abspath(os.path.join("data", "extracted", "images", "class_map.json"))
    if os.path.exists(cm_path):
        with open(cm_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = None
        print("⚠️ Warning: class_map.json not found. Will only show index.")

    # load model
    model = tf.keras.models.load_model("saved_models/pet_classifier.keras")

    # preprocess image
    img = load_image(image_path)
    preds = model.predict(img[None, ...])  # add batch dimension

    # extract prediction
    class_idx = preds.argmax(axis=-1)[0]
    prob = preds.max()

    # show result
    if class_names:
        label = class_names[class_idx]
        print(f"Predicted class: {class_idx} -> {label} (prob {prob:.3f})")
    else:
        print(f"Predicted class index: {class_idx} (prob {prob:.3f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
    else:
        main(sys.argv[1])
