import cv2
import numpy as np
import albumentations as A
import torch
from torchvision import transforms
import mediapipe as mp

# Albumentations transform pipeline for ViT model
def get_vit_transforms():
    """
    Returns the image augmentation and preprocessing pipeline using Albumentations.
    Used to preprocess images for the Vision Transformer (ViT) model.
    
    Returns:
        albumentations.Compose: Albumentations transformation pipeline.
    """
    return A.Compose([
        A.Resize(224, 224),  # Resize to 224x224 for ViT
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.RandomBrightnessContrast(p=0.5),  # Random brightness and contrast adjustments
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet normalization
        A.pytorch.transforms.ToTensorV2()  # Convert to tensor
    ])


# Preprocessing function for a single image for the ViT model
def preprocess_image_vit(image, vit_transforms):
    """
    Preprocess an image for the Vision Transformer (ViT) model using the defined Albumentations pipeline.
    
    Parameters:
        image (ndarray): Input image in BGR format (e.g., from cv2).
        vit_transforms (albumentations.Compose): Albumentations transformation pipeline.
    
    Returns:
        torch.Tensor: Preprocessed image tensor ready for the ViT model.
    """
    # Apply Albumentations transformations
    transformed = vit_transforms(image=image)['image']
    
    # Return the transformed image (now as a PyTorch tensor)
    return transformed


# MediaPipe initialization for LSTM keypoint extraction
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Keypoint extraction and preprocessing for the LSTM model
def preprocess_for_lstm(frame):
    """
    Preprocess a video frame to extract keypoints using MediaPipe's holistic model for LSTM model input.
    
    Parameters:
        frame (ndarray): Input video frame from webcam (BGR format).
    
    Returns:
        np.array: Flattened array of extracted keypoints for LSTM input.
    """
    # Convert the color space from BGR to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Disable writing to the image

    # Extract keypoints using the holistic model
    results = holistic.process(image)

    # Convert the image back to BGR for rendering (optional, only if you need to show it)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks from pose, face, and both hands
    pose = results.pose_landmarks.landmark if results.pose_landmarks else np.zeros(33 * 4)
    face = results.face_landmarks.landmark if results.face_landmarks else np.zeros(468 * 4)
    lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else np.zeros(21 * 4)
    rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else np.zeros(21 * 4)

    # Flatten the landmarks into a single vector
    keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in pose] +
                         [[res.x, res.y, res.z] for res in face] +
                         [[res.x, res.y, res.z] for res in lh] +
                         [[res.x, res.y, res.z] for res in rh]).flatten()

    # Return the keypoints as a flattened vector (if all landmarks are found, else return zeros)
    return keypoints if keypoints.shape == (1662,) else np.zeros(1662)


# General preprocessing function that decides which model's preprocessing to use
def preprocess_image_for_model(image, model_type):
    """
    Preprocess an image or frame based on the type of model being used (ViT or LSTM).
    
    Parameters:
        image (ndarray): Input image or frame from webcam.
        model_type (str): The type of model to preprocess the image for ('vit' or 'lstm').
    
    Returns:
        torch.Tensor or np.array: Preprocessed image or keypoints for the respective model.
    """
    if model_type == 'vit':
        vit_transforms = get_vit_transforms()  # Get ViT-specific transforms
        return preprocess_image_vit(image, vit_transforms)  # Preprocess for ViT
    elif model_type == 'lstm':
        return preprocess_for_lstm(image)  # Preprocess for LSTM
    else:
        raise ValueError("Unsupported model type. Choose either 'vit' or 'lstm'.")


if __name__ == "__main__":
    # Test the ViT and LSTM preprocessing functions
    vit_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)  # Simulated image for testing
    vit_preprocessed = preprocess_image_for_model(vit_image, 'vit')
    print(f"ViT Preprocessed Shape: {vit_preprocessed.shape}")
    
    # Simulate a webcam frame (replace with actual frame in production)
    lstm_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    lstm_preprocessed = preprocess_image_for_model(lstm_frame, 'lstm')
    print(f"LSTM Preprocessed Keypoints Shape: {lstm_preprocessed.shape}")
