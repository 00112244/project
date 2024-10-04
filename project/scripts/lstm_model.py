import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def load_lstm_model(model_path='model.h5'):
    """
    Load the pre-trained LSTM model for sign language detection.
    
    Parameters:
        model_path (str): Path to the saved LSTM model. Default is 'model.h5'.
        
    Returns:
        model: Loaded LSTM model ready for inference.
    """
    try:
        # Load the pre-trained model (compiled)
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded LSTM model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_for_lstm(frame, holistic, mp_drawing):
    """
    Preprocess a video frame to extract keypoints using MediaPipe's holistic model 
    and prepare the input for the LSTM model.
    
    Parameters:
        frame (ndarray): Input video frame from webcam.
        holistic: MediaPipe Holistic object for extracting keypoints.
        mp_drawing: MediaPipe drawing object for visualizing keypoints.
    
    Returns:
        np.array: Extracted keypoints in a flattened format for LSTM model input.
    """
    # Convert the color space from BGR to RGB (required for MediaPipe)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection using MediaPipe holistic model
    results = holistic.process(image)
    
    # Convert the color space back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks on the frame (optional for visualization)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    # Extract keypoints for both hands, pose, and face
    pose = results.pose_landmarks.landmark if results.pose_landmarks else np.zeros(33*4)
    face = results.face_landmarks.landmark if results.face_landmarks else np.zeros(468*4)
    lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else np.zeros(21*4)
    rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else np.zeros(21*4)
    
    # Combine keypoints into a single flattened array
    keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in pose] +
                         [[res.x, res.y, res.z] for res in face] +
                         [[res.x, res.y, res.z] for res in lh] +
                         [[res.x, res.y, res.z] for res in rh]).flatten()

    # Return the extracted keypoints (if all landmarks are found, else return zeros)
    return keypoints if keypoints.shape == (1662,) else np.zeros(1662)


def predict_with_lstm(model, keypoints):
    """
    Make a prediction using the LSTM model on the provided keypoints.
    
    Parameters:
        model (tf.keras.Model): Pre-trained LSTM model for sign language detection.
        keypoints (np.array): Flattened array of keypoints extracted from video.
    
    Returns:
        np.array: Model prediction (probabilities for each class).
    """
    keypoints = np.expand_dims(keypoints, axis=0)  # Add batch dimension
    predictions = model.predict(keypoints)
    return predictions


if __name__ == "__main__":
    # Test the model loading and prediction
    lstm_model = load_lstm_model('model.h5')
    
    if lstm_model:
        # Simulate keypoints input for testing (replace with actual data in production)
        dummy_keypoints = np.random.rand(1662)  # Fake keypoints for testing
        predictions = predict_with_lstm(lstm_model, dummy_keypoints)
        print(f"Predictions: {predictions}")
