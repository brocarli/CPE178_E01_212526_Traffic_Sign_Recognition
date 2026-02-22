import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


class TrafficSignModel:

    def __init__(self, input_shape=(48, 48, 3), num_classes=43,
                 use_transfer_learning=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_transfer_learning = use_transfer_learning
        self.model = None
        self.history = None

        print(f"Model Builder initialized")
        print(f"  Input shape: {input_shape}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Transfer learning: {use_transfer_learning}")

    def build_model(self):
        if self.use_transfer_learning:
            return self._build_transfer_learning_model()
        else:
            return self._build_custom_cnn()

    def _build_custom_cnn(self):
        print("\nBuilding enhanced CNN model with batch normalization...")

        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), padding='same',
                          input_shape=self.input_shape, name='conv1'),
            layers.BatchNormalization(name='bn1'),
            layers.Activation('relu', name='relu1'),
            layers.Conv2D(32, (3, 3), padding='same', name='conv2'),
            layers.BatchNormalization(name='bn2'),
            layers.Activation('relu', name='relu2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),

            # Block 2
            layers.Conv2D(64, (3, 3), padding='same', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            layers.Activation('relu', name='relu3'),
            layers.Conv2D(64, (3, 3), padding='same', name='conv4'),
            layers.BatchNormalization(name='bn4'),
            layers.Activation('relu', name='relu4'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),

            # Block 3
            layers.Conv2D(128, (3, 3), padding='same', name='conv5'),
            layers.BatchNormalization(name='bn5'),
            layers.Activation('relu', name='relu5'),
            layers.Conv2D(128, (3, 3), padding='same', name='conv6'),
            layers.BatchNormalization(name='bn6'),
            layers.Activation('relu', name='relu6'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),

            layers.Flatten(name='flatten'),

            layers.Dense(512, name='dense1'),
            layers.BatchNormalization(name='bn7'),
            layers.Activation('relu', name='relu7'),
            layers.Dropout(0.5, name='dropout4'),

            layers.Dense(256, name='dense2'),
            layers.BatchNormalization(name='bn8'),
            layers.Activation('relu', name='relu8'),
            layers.Dropout(0.5, name='dropout5'),

            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        print("✓ Enhanced CNN model built successfully!")
        print(f"\nModel Summary:")
        model.summary()

        return model

    def _build_transfer_learning_model(self):
        print("\nBuilding transfer learning model using MobileNetV2...")

        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        x = layers.Dense(256, name='dense1')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Activation('relu', name='relu1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax',
                               name='output')(x)

        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        print("✓ Transfer learning model built successfully!")
        print(f"  Base model: MobileNetV2 (frozen)")
        print(f"\nModel Summary:")
        model.summary()

        return model

    def fine_tune(self, unfreeze_layers=30):
        if self.model is None or not self.use_transfer_learning:
            print("Fine-tuning only applicable to transfer learning models.")
            return

        base_model = self.model.layers[1]
        base_model.trainable = True

        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        trainable_count = sum(1 for l in base_model.layers if l.trainable)
        print(f"\nFine-tuning: {trainable_count} layers unfrozen in base model")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✓ Model recompiled for fine-tuning with lr=1e-5")

    def train_model(self, X_train, y_train, X_val=None, y_val=None,
                    epochs=30, batch_size=32, validation_split=0.2):
        if self.model is None:
            print("Error: Model not built. Call build_model() first.")
            return None

        print("\n" + "=" * 50)
        print("Starting Model Training")
        print("=" * 50)
        print(f"Training samples: {len(X_train)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]

        if X_val is not None and y_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

        print("\n✓ Training complete!")
        return self.history

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            print("Error: Model not built.")
            return None

        print("\n" + "=" * 50)
        print("Evaluating Model")
        print("=" * 50)

        test_loss, test_accuracy = self.model.evaluate(X_test, y_test,
                                                        verbose=1)

        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

        return test_loss, test_accuracy

    def plot_training_history(self, save_path='training_history.png'):
        if self.history is None:
            print("No training history available.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'],
                 label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training history plot saved to {save_path}")
        plt.close()

    def save_model(self, filepath='traffic_sign_model.keras'):
        if self.model is None:
            print("Error: No model to save.")
            return

        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from {filepath}")


if __name__ == "__main__":

    print("=" * 50)
    print("Traffic Sign Recognition - Model Training (Sprint 2)")
    print("=" * 50)
    print()

    print("Creating sample dataset for demonstration...")
    X_sample = np.random.rand(100, 48, 48, 3).astype('float32')
    y_sample = np.random.randint(0, 43, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    model_builder = TrafficSignModel(input_shape=(48, 48, 3), num_classes=43,
                                      use_transfer_learning=False)

    model_builder.build_model()

    print("\nTraining model (demo with 2 epochs)...")
    history = model_builder.train_model(
        X_train, y_train,
        epochs=2,
        batch_size=16
    )

    model_builder.evaluate_model(X_test, y_test)

    model_builder.plot_training_history()

    model_builder.save_model('demo_model_sprint2.keras')

    print("\n✓ Demo complete!")
