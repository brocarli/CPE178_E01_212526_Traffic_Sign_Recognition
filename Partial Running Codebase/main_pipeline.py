import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

from data_preprocessing import DataPreprocessor
from data_augmentation import DataAugmentor
from model_training import TrafficSignModel
from utils import (visualize_images, check_data_distribution, 
                   save_training_config, create_project_summary)


def run_full_pipeline(data_folder='../Dataset_Sample', 
                      use_augmentation=True,
                      num_classes=43,
                      epochs=30,
                      batch_size=32):
    
    print("=" * 70)
    print("TRAFFIC SIGN RECOGNITION - COMPLETE PIPELINE")
    print("=" * 70)
    print()
    
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    preprocessor = DataPreprocessor(image_size=(32, 32))
    
    print(f"\nLoading data from: {data_folder}")
    images, labels = preprocessor.load_dataset_from_folder(data_folder)
    
    if len(images) == 0:
        print("\n❌ ERROR: No images loaded!")
        print("Please make sure your dataset is in the correct folder structure:")
        print("  Dataset_Sample/")
        print("    class_0/")
        print("      image1.jpg")
        print("    class_1/")
        print("      image1.jpg")
        print("    ...")
        return
    
    print("\nVisualizing sample images...")
    visualize_images(images[:25], labels[:25], 
                    save_path='results/sample_images.png')
    
    print("\nAnalyzing class distribution...")
    check_data_distribution(labels, 
                           save_path='results/original_distribution.png')
    
    if use_augmentation:
        print("\n" + "="*70)
        print("STEP 2: DATA AUGMENTATION")
        print("="*70)
        
        augmentor = DataAugmentor()
        
        images, labels = augmentor.augment_dataset(
            images, labels, 
            augmentations_per_image=2
        )
        
        print("\nAnalyzing augmented class distribution...")
        check_data_distribution(labels, 
                               save_path='results/augmented_distribution.png')
    
    print("\n" + "="*70)
    print("STEP 3: SPLITTING DATA INTO TRAIN/VALIDATION/TEST SETS")
    print("="*70)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, 
        test_size=0.2, 
        random_state=42,
        stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"\nDataset split complete:")
    print(f"  Training set:   {len(X_train):6d} samples ({len(X_train)/len(images)*100:.1f}%)")
    print(f"  Validation set: {len(X_val):6d} samples ({len(X_val)/len(images)*100:.1f}%)")
    print(f"  Test set:       {len(X_test):6d} samples ({len(X_test)/len(images)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("STEP 4: BUILDING AND TRAINING THE MODEL")
    print("="*70)
    
    model_builder = TrafficSignModel(
        input_shape=(32, 32, 3),
        num_classes=num_classes
    )
    
    model_builder.build_model()
    
    print("\nStarting training...")
    print("This may take a while depending on your hardware.")
    print("GPU will significantly speed up training.")
    
    history = model_builder.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    print("\nPlotting training history...")
    model_builder.plot_training_history(save_path='results/training_history.png')
    
    print("\n" + "="*70)
    print("STEP 5: EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    test_loss, test_accuracy = model_builder.evaluate_model(X_test, y_test)
    
    print("\n" + "="*70)
    print("STEP 6: SAVING MODEL AND RESULTS")
    print("="*70)
    
    model_builder.save_model('results/traffic_sign_model.keras')
    
    config = {
        'image_size': (32, 32, 3),
        'num_classes': num_classes,
        'augmentation_used': use_augmentation,
        'epochs': epochs,
        'batch_size': batch_size,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss)
    }
    save_training_config(config, 'results/training_config.json')
    
    create_project_summary(images, labels, model_builder.model,
                          save_path='results/project_summary.txt')
    
    print("\n" + "="*70)
    print("✓ PIPELINE EXECUTION COMPLETE!")
    print("="*70)
    print("\nResults saved in 'results/' folder:")
    print("  - traffic_sign_model.keras  : Trained model")
    print("  - training_history.png      : Training curves")
    print("  - sample_images.png         : Sample dataset images")
    print("  - training_config.json      : Training configuration")
    print("  - project_summary.txt       : Project summary")
    print()
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
    print()
    print("Next steps:")
    print("  1. Review the training history plot")
    print("  2. Test the model on new images")
    print("  3. Fine-tune hyperparameters if needed")
    print("  4. Add more data augmentation techniques")
    print("="*70)


def main():
    os.makedirs('results', exist_ok=True)
    
    CONFIG = {
        'data_folder': '../Dataset_Sample',
        'use_augmentation': True,
        'num_classes': 43,
        'epochs': 30,
        'batch_size': 32
    }
    
    try:
        run_full_pipeline(**CONFIG)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user.")
        print("Partial results may be saved in 'results/' folder.")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        print("\nPlease check:")
        print("  1. Dataset folder exists and has correct structure")
        print("  2. All required packages are installed")
        print("  3. Sufficient memory available")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
