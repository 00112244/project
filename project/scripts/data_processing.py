import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Define the path to your dataset
DATA_PATH = 'MP_Data'
PROCESSED_DATA_PATH = 'Processed_Data'

# Define the Albumentations transformations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.Resize(224, 224),  # Resize to the input size required by your model
    ToTensorV2()
])

def preprocess_image(image_path):
    """
    Load and preprocess an image using Albumentations.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        numpy.ndarray: Augmented and preprocessed image.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")

        # Convert the image to RGB (Albumentations works with RGB images)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        augmented = transform(image=image)
        image = augmented['image']
        
        return image
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_dataset(dataset_path):
    """
    Process all images in the dataset directory.
    
    Args:
        dataset_path (str): Path to the dataset directory.
    """
    for action_folder in tqdm(os.listdir(dataset_path), desc="Processing dataset"):
        action_path = os.path.join(dataset_path, action_folder)
        if not os.path.isdir(action_path):
            continue
        
        for seq_file in os.listdir(action_path):
            seq_path = os.path.join(action_path, seq_file)
            if not seq_file.endswith('.jpg'):
                continue
            
            # Preprocess each image
            processed_image = preprocess_image(seq_path)
            if processed_image is None:
                continue
            
            # Save the processed image
            save_path = os.path.join(PROCESSED_DATA_PATH, action_folder, seq_file)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    process_dataset(DATA_PATH)
