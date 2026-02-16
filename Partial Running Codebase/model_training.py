import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


class TrafficSignModel:
    
    def __init__(self, input_shape=(32, 32, 3), num_classes=43):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        print(f"Model Builder initialized")
        print(f"  Input shape: {input_shape}")
        print(f"  Number of classes: {num_classes}")
    
    def build_model(self):
        print("\nBuilding CNN model...")
        
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=self.input_shape, 
                         padding='same',
                         name='conv1'),
            layers.Conv2D(32, (3, 3), activation='relu', 
                         padding='same',
                         name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            layers.Conv2D(64, (3, 3), activation='relu', 
                         padding='same',
                         name='conv3'),
            layers.Conv2D(64, (3, 3), activation='relu', 
                         padding='same',
                         name='conv4'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            layers.Conv2D(128, (3, 3), activation='relu', 
                         padding='same',
                         name='conv5'),
            layers.Conv2D(128, (3, 3), activation='relu', 
                         padding='same',
                         name='conv6'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),
            
            layers.Flatten(name='flatten'),
            
            layers.Dense(512, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout4'),
            layers.Dense(256, activation='relu', name='dense2'),
            layers.Dropout(0.5, name='dropout5'),
            
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("✓ Model built successfully!")
        print(f"\nModel Summary:")
        model.summary()
        
        return model
    
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
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        
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
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
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
    print("Traffic Sign Recognition - Model Training")
    print("=" * 50)
    print()
    
    print("Creating sample dataset for demonstration...")
    X_sample = np.random.rand(100, 32, 32, 3).astype('float32')
    y_sample = np.random.randint(0, 43, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    model_builder = TrafficSignModel(num_classes=43)
    
    model_builder.build_model()
    
    print("\nTraining model (demo with 2 epochs)...")
    history = model_builder.train_model(
        X_train, y_train,
        epochs=2,
        batch_size=16
    )
    
    model_builder.evaluate_model(X_test, y_test)
    
    model_builder.plot_training_history()
    
    model_builder.save_model('demo_model.keras')
    
    print("\n✓ Demo complete!")
