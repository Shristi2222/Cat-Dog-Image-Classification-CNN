"""
=========================================================
Cat vs Dog Image Classification - Training Script
Author: Shristi Bastola
=========================================================
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt

# =====================================================
# Configuration
# =====================================================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = r"C:\Users\Dell\OneDrive\Desktop\practice\CNN_project\dataset\dogs-vs-cats-classification"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALIDATION_PATH = os.path.join(DATASET_PATH, "validation")
TEST_PATH = os.path.join(DATASET_PATH, "test")

MODEL_FOLDER = r"C:\Users\Dell\OneDrive\Desktop\practice\CNN_project\models"
RESULT_FOLDER = r"C:\Users\Dell\OneDrive\Desktop\practice\CNN_project\results"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

print("Loading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("\nClasses:", train_ds.class_names)

# =====================================================
# Normalize Images
# =====================================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =====================================================
# CNN Model
# =====================================================

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE,3)
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )

])

model.summary()

# =====================================================
# Compile Model
# =====================================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

# =====================================================
# Callbacks
# =====================================================

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    r"C:\Users\Dell\OneDrive\Desktop\practice\CNN_project\models\best_model.keras",

    save_best_only=True

)

# =====================================================
# Train Model
# =====================================================

print("\nTraining Started...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[early_stop, checkpoint]

)

# =====================================================
# Evaluate Model
# =====================================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(test_ds)

print(f"\nTest Accuracy : {accuracy:.4f}")

print(f"Test Loss     : {loss:.4f}")

# =====================================================
# Save Final Model
# =====================================================

model.save("../models/dog_cat_model.keras")

print("\nModel Saved Successfully!")

# ============================================
# Accuracy Plot
# ============================================

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(r"C:\Users\Dell\OneDrive\Desktop\practice\CNN_project\results\accuracy.png",
            dpi=300,
            bbox_inches="tight")

plt.close()


# ============================================
# Loss Plot
# ============================================

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(r"C:\Users\Dell\OneDrive\Desktop\practice\CNN_project\results\loss.png",
            dpi=300,
            bbox_inches="tight")

plt.close()

print("Both plots saved successfully!")