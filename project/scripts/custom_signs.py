import os
import numpy as np
import json
import cv2
from preprocessing import preprocess_for_lstm  # Use the LSTM preprocessing to extract keypoints

# Define directories to store custom signs data
CUSTOM_SIGNS_DIR = 'custom_signs'  # Directory to save custom sign keypoints
if not os.path.exists(CUSTOM_SIGNS_DIR):
    os.makedirs(CUSTOM_SIGNS_DIR)

def save_custom_sign(sign_name, keypoints):
    """
    Save the extracted keypoints of a custom sign to a file for future detection.
    
    Parameters:
        sign_name (str): Name of the custom sign.
        keypoints (np.array): Extracted keypoints from the sign (LSTM input).
    
    Returns:
        bool: True if sign is saved successfully, False otherwise.
    """
    try:
        # Create a directory for the sign if it doesn't exist
        sign_dir = os.path.join(CUSTOM_SIGNS_DIR, sign_name)
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)
        
        # Save keypoints as a JSON file for each sign instance
        keypoints_path = os.path.join(sign_dir, f'{sign_name}_keypoints.json')
        with open(keypoints_path, 'w') as f:
            json.dump(keypoints.tolist(), f)  # Convert numpy array to list before saving
        print(f"Custom sign '{sign_name}' saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving custom sign '{sign_name}': {e}")
        return False


def load_custom_sign(sign_name):
    """
    Load the keypoints of a previously saved custom sign for detection.
    
    Parameters:
        sign_name (str): Name of the custom sign.
    
    Returns:
        np.array: Loaded keypoints for the custom sign, or None if loading fails.
    """
    try:
        # Load keypoints from the JSON file
        sign_dir = os.path.join(CUSTOM_SIGNS_DIR, sign_name)
        keypoints_path = os.path.join(sign_dir, f'{sign_name}_keypoints.json')
        with open(keypoints_path, 'r') as f:
            keypoints = np.array(json.load(f))  # Convert the list back to a numpy array
        print(f"Custom sign '{sign_name}' loaded successfully.")
        return keypoints
    except Exception as e:
        print(f"Error loading custom sign '{sign_name}': {e}")
        return None


def detect_custom_sign(lstm_model, current_keypoints, custom_sign_keypoints):
    """
    Detect whether the current keypoints match the stored custom sign keypoints.
    
    Parameters:
        lstm_model: The trained LSTM model for sign language detection.
        current_keypoints (np.array): Extracted keypoints from the current frame.
        custom_sign_keypoints (np.array): Pre-stored keypoints for the custom sign.
    
    Returns:
        bool: True if the custom sign is detected, False otherwise.
    """
    # Normalize keypoints before comparison (optional based on your preprocessing)
    if current_keypoints.shape != custom_sign_keypoints.shape:
        return False
    
    # Run the current keypoints through the LSTM model (if you want to classify via model)
    # For now, let's simply use Euclidean distance as a simple metric for matching.
    distance = np.linalg.norm(current_keypoints - custom_sign_keypoints)
    
    # Tune this threshold based on testing. A smaller threshold means stricter matching.
    threshold = 10  # Adjust this threshold to make matching more or less strict
    
    if distance < threshold:
        return True
    return False


def record_custom_sign(sign_name, num_frames=30):
    """
    Record a custom sign by capturing keypoints over a number of frames using the webcam.
    
    Parameters:
        sign_name (str): The name to save the custom sign under.
        num_frames (int): Number of frames to capture for sign recording.
    
    Returns:
        bool: True if the custom sign was recorded successfully, False otherwise.
    """
    cap = cv2.VideoCapture(0)
    keypoints_list = []

    print(f"Recording custom sign '{sign_name}'. Please perform the sign in front of the camera...")

    try:
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                print("Error capturing video frame.")
                break

            # Extract keypoints from the current frame using the LSTM preprocessing method
            keypoints = preprocess_for_lstm(frame)
            
            # If valid keypoints are detected, save them
            if keypoints is not None and len(keypoints) > 0:
                keypoints_list.append(keypoints)
            
            # Display the frame for visual feedback
            cv2.imshow(f"Recording '{sign_name}'", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(keypoints_list) == 0:
            print(f"No keypoints detected for sign '{sign_name}'. Recording failed.")
            return False

        # Average the keypoints over the recorded frames (optional, for consistency)
        averaged_keypoints = np.mean(keypoints_list, axis=0)

        # Save the averaged keypoints as the custom sign
        return save_custom_sign(sign_name, averaged_keypoints)
    except Exception as e:
        print(f"Error recording custom sign '{sign_name}': {e}")
        return False


if __name__ == "__main__":
    # Example: Record a new custom sign
    sign_name = "my_custom_sign"
    if record_custom_sign(sign_name):
        print(f"Custom sign '{sign_name}' recorded and saved successfully.")

    # Example: Load and detect a custom sign
    lstm_model = None  # Load your pre-trained LSTM model here
    current_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Simulated frame for testing
    current_keypoints = preprocess_for_lstm(current_frame)
    custom_sign_keypoints = load_custom_sign(sign_name)
    
    if custom_sign_keypoints is not None:
        is_detected = detect_custom_sign(lstm_model, current_keypoints, custom_sign_keypoints)
        if is_detected:
            print(f"Custom sign '{sign_name}' detected!")
        else:
            print(f"Custom sign '{sign_name}' not detected.")
