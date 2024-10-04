import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

def load_vit_model(model_path='best_vit_model.pth', num_classes=27):
    """
    Load the pre-trained Vision Transformer (ViT) model for sign language detection.
    
    Parameters:
        model_path (str): Path to the saved ViT model weights.
        num_classes (int): Number of sign language classes. Default is 27.
        
    Returns:
        model: Loaded and configured ViT model.
    """
    try:
        # Load pre-trained ViT model from torchvision and modify the classification head
        model = models.vit_b_16(pretrained=True)
        model.heads = torch.nn.Linear(model.heads.in_features, num_classes)
        
        # Load the custom weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        print(f"Loaded ViT model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        return None


def preprocess_for_vit(frame):
    """
    Preprocess the video frame for ViT model input.
    
    Parameters:
        frame (ndarray): Input video frame from webcam (BGR format).
    
    Returns:
        torch.Tensor: Preprocessed image tensor ready for ViT model inference.
    """
    # Define the image preprocessing pipeline (resize, normalize, etc.)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to the ViT model's expected input size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize to ImageNet values
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Apply preprocessing to the input frame
    input_tensor = preprocess(frame)
    
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
    
    return input_tensor


def predict_with_vit(model, input_tensor):
    """
    Make a prediction using the ViT model on the preprocessed input.
    
    Parameters:
        model (torch.nn.Module): Pre-trained ViT model.
        input_tensor (torch.Tensor): Preprocessed image tensor.
    
    Returns:
        np.array: Model prediction (probabilities for each class).
    """
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(input_tensor)  # Forward pass
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Get class probabilities
        predictions = probabilities.numpy()  # Convert to numpy array
    
    return predictions


if __name__ == "__main__":
    # Test the model loading and prediction
    vit_model = load_vit_model('best_vit_model.pth')
    
    if vit_model:
        # Simulate an input image for testing (replace with actual data in production)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)  # Fake image for testing
        input_tensor = preprocess_for_vit(dummy_image)
        predictions = predict_with_vit(vit_model, input_tensor)
        print(f"Predictions: {predictions}")
