import numpy as np
import cv2
import os
from pathlib import Path


class DataPreprocessor:

    def __init__(self, image_size=(48, 48)):
        self.image_size = image_size
        print(f"Data Preprocessor initialized with image size: {image_size}")

    def load_image(self, image_path):
        try:
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Warning: Could not load image {image_path}")
                return None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def preprocess_image(self, image):
        resized = cv2.resize(image, self.image_size)
        normalized = resized.astype('float32') / 255.0
        return normalized

    def load_dataset_from_folder(self, data_folder, limit=None):
        images = []
        labels = []

        data_path = Path(data_folder)

        if not data_path.exists():
            print(f"Error: Folder {data_folder} does not exist!")
            return np.array([]), np.array([])

        class_folders = sorted([f for f in data_path.iterdir() if f.is_dir()])

        print(f"Found {len(class_folders)} classes")

        for class_idx, class_folder in enumerate(class_folders):
            print(f"Loading class {class_idx}: {class_folder.name}")

            image_files = list(class_folder.glob('*.jpg')) + \
                         list(class_folder.glob('*.png')) + \
                         list(class_folder.glob('*.ppm'))

            if limit:
                image_files = image_files[:limit]

            for img_path in image_files:
                image = self.load_image(img_path)

                if image is not None:
                    preprocessed = self.preprocess_image(image)
                    images.append(preprocessed)
                    labels.append(class_idx)

            print(f"  Loaded {len(image_files)} images")

        images = np.array(images)
        labels = np.array(labels)

        print(f"\nTotal images loaded: {len(images)}")
        print(f"Image shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")

        return images, labels

    def save_preprocessed_data(self, images, labels, output_file):
        np.savez(output_file, images=images, labels=labels)
        print(f"Data saved to {output_file}")

    def load_preprocessed_data(self, input_file):
        data = np.load(input_file)
        images = data['images']
        labels = data['labels']
        print(f"Loaded {len(images)} images from {input_file}")
        return images, labels


if __name__ == "__main__":

    print("=" * 50)
    print("Traffic Sign Recognition - Data Preprocessing")
    print("=" * 50)
    print()

    preprocessor = DataPreprocessor(image_size=(48, 48))

    data_folder = "../Dataset_Sample"

    print(f"Attempting to load data from: {data_folder}")
    images, labels = preprocessor.load_dataset_from_folder(data_folder, limit=10)

    if len(images) > 0:
        print("\n✓ Data loaded successfully!")
        print(f"  Image array shape: {images.shape}")
        print(f"  Label array shape: {labels.shape}")
        print(f"  Number of unique classes: {len(np.unique(labels))}")

        preprocessor.save_preprocessed_data(images, labels, "preprocessed_data.npz")
    else:
        print("\n✗ No data loaded. Please check your dataset folder.")
        print("  Make sure images are organized in class folders.")
