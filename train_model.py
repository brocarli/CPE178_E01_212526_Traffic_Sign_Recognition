"""
==============================================
  TRAFFIC SIGN RECOGNITION - TRAINING SCRIPT
==============================================

HOW YOUR DATASET SHOULD BE ORGANIZED:
--------------------------------------
dataset/
  ├── stop_sign/
  │     ├── image1.jpg
  │     ├── image2.jpg
  │     └── ...
  ├── speed_limit/
  │     ├── image1.jpg
  │     └── ...
  └── yield/
        └── ...

Each FOLDER = one type of traffic sign.
The folder NAME = the label/class name.

HOW TO RUN THIS SCRIPT:
------------------------
1. Install requirements:
   pip install tensorflow pillow numpy scikit-learn

2. Put your dataset in the same folder as this script.

3. Run:
   python train_model.py

4. Wait for training to finish. It will create:
   - traffic_sign_model.h5  (the trained AI model)
   - class_names.txt        (list of your sign categories)
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# ── SETTINGS (you can change these) ──────────────────────────────────────────
DATASET_FOLDER = "dataset"   # folder where your images are
IMAGE_SIZE     = (64, 64)    # all images will be resized to this
EPOCHS         = 15          # how many times the AI trains over the data
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(dataset_folder):
    """Reads all images from the dataset folder."""
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_folder))  # folder names = class names
    class_names = [c for c in class_names if os.path.isdir(os.path.join(dataset_folder, c))]

    print(f"\n✅ Found {len(class_names)} categories: {class_names}\n")

    for label_index, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_folder, class_name)
        for image_file in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_file)
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize(IMAGE_SIZE)
                images.append(np.array(img))
                labels.append(label_index)
            except Exception:
                pass  # skip unreadable files

    images = np.array(images, dtype="float32") / 255.0  # normalize to 0-1
    labels = np.array(labels)
    return images, labels, class_names


def build_model(num_classes):
    """Builds a simple CNN model."""
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        # --- Feature extraction layers ---
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        # --- Classification layers ---
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def main():
    print("=" * 50)
    print("  TRAFFIC SIGN RECOGNITION - MODEL TRAINER")
    print("=" * 50)

    # 1. Load data
    print("\n📂 Loading images from dataset...")
    if not os.path.exists(DATASET_FOLDER):
        print(f"\n❌ ERROR: Could not find '{DATASET_FOLDER}' folder.")
        print("   Make sure your dataset folder is in the same location as this script.")
        return

    images, labels, class_names = load_dataset(DATASET_FOLDER)
    print(f"   Loaded {len(images)} images total.")

    # 2. Save class names
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"   Saved class names to class_names.txt")

    # 3. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    print(f"   Training images: {len(X_train)}")
    print(f"   Testing images:  {len(X_test)}")

    # 4. Build and compile model
    print("\n🧠 Building AI model...")
    model = build_model(num_classes=len(class_names))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # 5. Train the model
    print(f"\n🚀 Training for {EPOCHS} epochs...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        batch_size=32,
    )

    # 6. Evaluate
    print("\n📊 Evaluating on test images...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"   Test Accuracy: {accuracy * 100:.2f}%")

    # 7. Save the model
    model.save("traffic_sign_model.h5")
    print("\n✅ Model saved as: traffic_sign_model.h5")
    print("✅ Class names saved as: class_names.txt")
    print("\n🎉 Done! You can now run app.py to use the recognizer.")


if __name__ == "__main__":
    main()
