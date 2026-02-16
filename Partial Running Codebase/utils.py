import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json


def visualize_images(images, labels, class_names=None, num_images=25, 
                     save_path='sample_images.png'):
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_images):
        if i >= len(images):
            break
        
        img = images[i]
        label = labels[i]
        
        axes[i].imshow(img)
        
        if class_names and label < len(class_names):
            title = f"Class {label}: {class_names[label]}"
        else:
            title = f"Class {label}"
        
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
    
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}")
    plt.close()


def check_data_distribution(labels, class_names=None, save_path='class_distribution.png'):
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(14, 6))
    plt.bar(unique, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Dataset')
    plt.xticks(unique, rotation=90)
    plt.grid(axis='y', alpha=0.3)
    
    for i, (cls, count) in enumerate(zip(unique, counts)):
        plt.text(cls, count, str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Class distribution plot saved to {save_path}")
    plt.close()
    
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Number of classes: {len(unique)}")
    print(f"  Min samples per class: {counts.min()}")
    print(f"  Max samples per class: {counts.max()}")
    print(f"  Average samples per class: {counts.mean():.2f}")
    print(f"  Standard deviation: {counts.std():.2f}")
    
    if counts.max() / counts.min() > 5:
        print("\n⚠ Warning: Significant class imbalance detected!")
        print("  Consider using data augmentation or class weights.")


def predict_single_image(model, image, class_names=None, top_k=5):
    image_batch = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image_batch, verbose=0)[0]
    
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    print("\nPrediction Results:")
    print("-" * 40)
    for i, idx in enumerate(top_indices, 1):
        prob = predictions[idx] * 100
        if class_names and idx < len(class_names):
            name = class_names[idx]
            print(f"{i}. Class {idx} ({name}): {prob:.2f}%")
        else:
            print(f"{i}. Class {idx}: {prob:.2f}%")
    
    predicted_class = top_indices[0]
    confidence = predictions[predicted_class]
    
    return predicted_class, confidence


def create_confusion_matrix(model, X_test, y_test, class_names=None,
                           save_path='confusion_matrix.png'):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def save_training_config(config_dict, filepath='training_config.json'):
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Configuration saved to {filepath}")


def load_training_config(filepath='training_config.json'):
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"✓ Configuration loaded from {filepath}")
    return config


def calculate_model_size(model):
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    size_mb = (total_params * 4) / (1024 * 1024)
    
    print("\nModel Size Information:")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {non_trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Estimated size: {size_mb:.2f} MB")


def create_project_summary(images, labels, model=None, save_path='project_summary.txt'):
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAFFIC SIGN RECOGNITION PROJECT - SPRINT 1 SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total images: {len(images)}\n")
        f.write(f"Image shape: {images.shape}\n")
        f.write(f"Number of classes: {len(np.unique(labels))}\n")
        f.write(f"Labels shape: {labels.shape}\n\n")
        
        unique, counts = np.unique(labels, return_counts=True)
        f.write(f"Samples per class:\n")
        f.write(f"  Min: {counts.min()}\n")
        f.write(f"  Max: {counts.max()}\n")
        f.write(f"  Mean: {counts.mean():.2f}\n")
        f.write(f"  Std: {counts.std():.2f}\n\n")
        
        if model:
            f.write("MODEL INFORMATION\n")
            f.write("-" * 60 + "\n")
            
            trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
            f.write(f"Total parameters: {trainable:,}\n")
            f.write(f"Model architecture: CNN with {len(model.layers)} layers\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("End of Summary\n")
        f.write("=" * 60 + "\n")
    
    print(f"✓ Project summary saved to {save_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("Traffic Sign Recognition - Utility Functions")
    print("=" * 50)
    print()
    
    print("Creating sample data...")
    sample_images = np.random.rand(50, 32, 32, 3).astype('float32')
    sample_labels = np.random.randint(0, 10, 50)
    
    print("\n1. Visualizing sample images...")
    visualize_images(sample_images, sample_labels, num_images=16)
    
    print("\n2. Checking class distribution...")
    check_data_distribution(sample_labels)
    
    print("\n3. Creating project summary...")
    create_project_summary(sample_images, sample_labels)
    
    print("\n✓ All utility functions demonstrated!")
