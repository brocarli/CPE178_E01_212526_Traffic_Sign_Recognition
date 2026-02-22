import numpy as np
import cv2
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras


class TrafficSignPredictor:

    # GTSRB class names (German Traffic Sign Recognition Benchmark)
    CLASS_NAMES = [
        'Speed limit (20km/h)',
        'Speed limit (30km/h)',
        'Speed limit (50km/h)',
        'Speed limit (60km/h)',
        'Speed limit (70km/h)',
        'Speed limit (80km/h)',
        'End of speed limit (80km/h)',
        'Speed limit (100km/h)',
        'Speed limit (120km/h)',
        'No passing',
        'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection',
        'Priority road',
        'Yield',
        'Stop',
        'No vehicles',
        'Vehicles over 3.5 metric tons prohibited',
        'No entry',
        'General caution',
        'Dangerous curve to the left',
        'Dangerous curve to the right',
        'Double curve',
        'Bumpy road',
        'Slippery road',
        'Road narrows on the right',
        'Road work',
        'Traffic signals',
        'Pedestrians',
        'Children crossing',
        'Bicycles crossing',
        'Beware of ice/snow',
        'Wild animals crossing',
        'End of all speed and passing limits',
        'Turn right ahead',
        'Turn left ahead',
        'Ahead only',
        'Go straight or right',
        'Go straight or left',
        'Keep right',
        'Keep left',
        'Roundabout mandatory',
        'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons',
    ]

    def __init__(self, model_path=None, image_size=(48, 48)):
        self.image_size = image_size
        self.model = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False

        self.model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        return True

    def preprocess_image(self, image):
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                print(f"Error: Could not load image from {image}")
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()

        img = cv2.resize(img, self.image_size)
        img = img.astype('float32') / 255.0
        return img

    def predict_single(self, image, top_k=5):
        if self.model is None:
            print("Error: No model loaded. Call load_model() first.")
            return None, None

        preprocessed = self.preprocess_image(image)
        if preprocessed is None:
            return None, None

        image_batch = np.expand_dims(preprocessed, axis=0)
        predictions = self.model.predict(image_batch, verbose=0)[0]

        top_indices = np.argsort(predictions)[-top_k:][::-1]

        print("\nPrediction Results:")
        print("-" * 50)
        for rank, idx in enumerate(top_indices, 1):
            prob = predictions[idx] * 100
            name = self.CLASS_NAMES[idx] if idx < len(self.CLASS_NAMES) else f"Class {idx}"
            print(f"  {rank}. Class {idx:2d} - {name:<45s}: {prob:6.2f}%")

        predicted_class = int(top_indices[0])
        confidence = float(predictions[predicted_class])
        predicted_name = (self.CLASS_NAMES[predicted_class]
                          if predicted_class < len(self.CLASS_NAMES)
                          else f"Class {predicted_class}")

        print(f"\n  ✓ Predicted: {predicted_name} (confidence: {confidence*100:.2f}%)")

        return predicted_class, confidence

    def predict_batch(self, images):
        if self.model is None:
            print("Error: No model loaded. Call load_model() first.")
            return None, None

        preprocessed = []
        for img in images:
            p = self.preprocess_image(img)
            if p is not None:
                preprocessed.append(p)

        if not preprocessed:
            print("Error: No valid images to predict.")
            return None, None

        batch = np.array(preprocessed)
        predictions = self.model.predict(batch, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        print(f"\n✓ Batch prediction complete: {len(predicted_classes)} images")
        return predicted_classes, confidences

    def predict_from_folder(self, folder_path, top_k=3):
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Error: Folder {folder_path} does not exist!")
            return {}

        image_files = (list(folder.glob('*.jpg')) +
                       list(folder.glob('*.png')) +
                       list(folder.glob('*.ppm')))

        if not image_files:
            print(f"No images found in {folder_path}")
            return {}

        print(f"\nPredicting {len(image_files)} images from {folder_path}")
        print("=" * 60)

        results = {}
        for img_path in image_files:
            print(f"\nImage: {img_path.name}")
            predicted_class, confidence = self.predict_single(img_path, top_k=top_k)
            if predicted_class is not None:
                results[str(img_path)] = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_name': (self.CLASS_NAMES[predicted_class]
                                   if predicted_class < len(self.CLASS_NAMES)
                                   else f"Class {predicted_class}")
                }

        return results


if __name__ == "__main__":

    print("=" * 50)
    print("Traffic Sign Recognition - Prediction (Sprint 2)")
    print("=" * 50)
    print()

    print("Demonstrating predictor with random data...")
    print("(In practice, provide a path to a trained model)")

    predictor = TrafficSignPredictor(image_size=(48, 48))

    print("\nClass names available:")
    for i, name in enumerate(TrafficSignPredictor.CLASS_NAMES):
        print(f"  Class {i:2d}: {name}")

    print("\n✓ Predictor demo complete!")
    print("  To use: predictor = TrafficSignPredictor('path/to/model.keras')")
    print("  Then:   predictor.predict_single('path/to/image.jpg')")
