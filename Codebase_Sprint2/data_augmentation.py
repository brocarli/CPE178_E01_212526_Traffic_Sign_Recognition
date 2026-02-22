import numpy as np
import cv2
from scipy import ndimage
import random


class DataAugmentor:
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        print("Data Augmentor initialized")
    
    def rotate_image(self, image, max_angle=15):
        angle = random.uniform(-max_angle, max_angle)
        rotated = ndimage.rotate(image, angle, reshape=False, mode='nearest')
        return rotated
    
    def adjust_brightness(self, image, factor=None):
        if factor is None:
            factor = random.uniform(0.5, 1.5)
        adjusted = image * factor
        adjusted = np.clip(adjusted, 0, 1)
        return adjusted
    
    def add_noise(self, image, noise_level=0.01):
        noise = np.random.normal(0, noise_level, image.shape)
        noisy = image + noise
        noisy = np.clip(noisy, 0, 1)
        return noisy
    
    def shift_image(self, image, max_shift=4):
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        rows, cols = image.shape[:2]
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (cols, rows))
        return shifted
    
    def zoom_image(self, image, zoom_range=(0.9, 1.1)):
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        height, width = image.shape[:2]
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)
        zoomed = cv2.resize(image, (new_width, new_height))
        
        if zoom_factor > 1:
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            zoomed = zoomed[start_y:start_y+height, start_x:start_x+width]
        else:
            pad_y = (height - new_height) // 2
            pad_x = (width - new_width) // 2
            zoomed = cv2.copyMakeBorder(
                zoomed, pad_y, height-new_height-pad_y,
                pad_x, width-new_width-pad_x,
                cv2.BORDER_REPLICATE
            )
        
        return zoomed
    
    def augment_image(self, image, num_augmentations=5):
        augmented_images = []
        
        for i in range(num_augmentations):
            aug_img = image.copy()
            
            if random.random() > 0.3:
                aug_img = self.rotate_image(aug_img)
            
            if random.random() > 0.3:
                aug_img = self.adjust_brightness(aug_img)
            
            if random.random() > 0.3:
                aug_img = self.add_noise(aug_img)
            
            if random.random() > 0.3:
                aug_img = self.shift_image(aug_img)
            
            if random.random() > 0.3:
                aug_img = self.zoom_image(aug_img)
            
            augmented_images.append(aug_img)
        
        return augmented_images
    
    def augment_dataset(self, images, labels, augmentations_per_image=3):
        print(f"Starting data augmentation...")
        print(f"Original dataset size: {len(images)} images")
        print(f"Creating {augmentations_per_image} augmentations per image")
        
        augmented_images = list(images)
        augmented_labels = list(labels)
        
        for idx, (image, label) in enumerate(zip(images, labels)):
            aug_imgs = self.augment_image(image, augmentations_per_image)
            augmented_images.extend(aug_imgs)
            augmented_labels.extend([label] * len(aug_imgs))
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(images)} images")
        
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        
        print(f"✓ Augmentation complete!")
        print(f"  Final dataset size: {len(augmented_images)} images")
        print(f"  Increase: {len(augmented_images) - len(images)} new images")
        
        return augmented_images, augmented_labels


if __name__ == "__main__":
    
    print("=" * 50)
    print("Traffic Sign Recognition - Data Augmentation")
    print("=" * 50)
    print()
    
    augmentor = DataAugmentor()
    
    print("Creating sample image for demonstration...")
    sample_image = np.random.rand(32, 32, 3).astype('float32')
    
    print("\nGenerating 5 augmented versions...")
    augmented = augmentor.augment_image(sample_image, num_augmentations=5)
    print(f"✓ Created {len(augmented)} augmented images")
    
    print("\n" + "=" * 50)
    print("Example: Augmenting a small dataset")
    print("=" * 50)
    
    sample_images = np.random.rand(10, 32, 32, 3).astype('float32')
    sample_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    aug_images, aug_labels = augmentor.augment_dataset(
        sample_images, 
        sample_labels, 
        augmentations_per_image=3
    )
    
    print(f"\nFinal results:")
    print(f"  Original: {len(sample_images)} images")
    print(f"  Augmented: {len(aug_images)} images")
    print(f"  Class distribution: {np.bincount(aug_labels)}")
