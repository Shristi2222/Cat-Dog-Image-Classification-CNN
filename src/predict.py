"""
=========================================================
Cat vs Dog Image Classification - Prediction Script
Author: Shristi Bastola
=========================================================
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# =====================================================
# Configuration
# =====================================================

IMG_SIZE = 128

MODEL_PATH = r"C:\Users\Dell\OneDrive\Desktop\practice\CNN_project\models\best_model.keras"

# =====================================================
# Load Model
# =====================================================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!\n")

# =====================================================
# Get Image Path
# =====================================================

image_path = input("Enter the full path of the image: ").strip()

if not os.path.exists(image_path):
    print("\nError: Image not found.")
    exit()

# =====================================================
# Load Image
# =====================================================

img = image.load_img(
    image_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# =====================================================
# Predict
# =====================================================

print("Image loaded successfully.")
print("Starting prediction...")

prediction = model.predict(img_array, verbose=1)

print("Prediction completed.")
print("Raw prediction:", prediction)

probability = prediction[0][0]


if probability >= 0.5:
    label = "DOG"
    confidence = probability * 100
else:
    label = "CAT"
    confidence = (1 - probability) * 100

# =====================================================
# Display Result
# =====================================================

print("\n===============================")
print("Prediction Result")
print("===============================")

print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}%")

# =====================================================
# Show Image
# =====================================================

plt.imshow(img)

plt.axis("off")

plt.title(f"{label}\nConfidence: {confidence:.2f}%")

plt.show()