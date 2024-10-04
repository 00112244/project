from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from lstm_model import load_lstm_model, predict_with_lstm
from vit_model import load_vit_model, predict_with_vit
from custom_signs import save_custom_sign, load_custom_sign, record_custom_sign, detect_custom_sign
from preprocessing import preprocess_for_lstm, preprocess_for_vit
import os

# Update the paths for static and templates
app = Flask(__name__, static_folder='web/static', template_folder='web/templates')

# Load the pre-trained models
lstm_model = load_lstm_model('models/model.h5')
vit_model = load_vit_model('models/best_vit_model.pth')

# Global variable for custom sign detection
current_custom_sign_name = None


@app.route('/')
def index():
    # Main index page
    return render_template('index.html')


@app.route('/detect')
def detect():
    # Page for real-time detection from dataset
    return render_template('detect.html')


@app.route('/custom')
def custom():
    # Page for adding and detecting custom signs
    return render_template('custom.html')


def generate_video_stream(model_type):
    """ Generate video stream for real-time sign detection. """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for model processing
        frame_resized = cv2.resize(frame, (640, 480))

        if model_type == 'lstm':
            # Preprocess the frame for LSTM and predict using the LSTM model
            keypoints = preprocess_for_lstm(frame_resized)
            if keypoints is not None:
                predicted_sign = predict_with_lstm(lstm_model, keypoints)
            else:
                predicted_sign = "No sign detected"

        elif model_type == 'vit':
            # Preprocess the frame for ViT and predict using the ViT model
            processed_frame = preprocess_for_vit(frame_resized)
            predicted_sign = predict_with_vit(vit_model, processed_frame)
        
        # Display predicted sign on the frame
        cv2.putText(frame, predicted_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Encode frame and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<model_type>')
def video_feed(model_type):
    """ Route for video streaming. """
    return Response(generate_video_stream(model_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/record_custom_sign', methods=['POST'])
def record_custom():
    """ Endpoint to record a custom sign. """
    sign_name = request.form.get('sign_name', '').strip()
    
    if not sign_name:
        return jsonify({"status": "failed", "message": "Sign name is required"}), 400

    try:
        # Record the custom sign using the webcam
        result = record_custom_sign(sign_name)
        if result:
            return jsonify({"status": "success", "message": f"Custom sign '{sign_name}' recorded successfully!"}), 200
        else:
            return jsonify({"status": "failed", "message": f"Failed to record custom sign '{sign_name}'"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/detect_custom_sign', methods=['POST'])
def detect_custom():
    """ Endpoint to detect a custom sign. """
    sign_name = request.form['sign_name']
    
    if not sign_name:
        return jsonify({"status": "failed", "message": "Sign name is required"}), 400

    # Load the custom sign keypoints
    custom_sign_keypoints = load_custom_sign(sign_name)
    
    if custom_sign_keypoints is None:
        return jsonify({"status": "failed", "message": f"Custom sign '{sign_name}' not found"}), 404

    # Set the global variable for detecting the custom sign in the video stream
    global current_custom_sign_name
    current_custom_sign_name = sign_name

    return jsonify({"status": "success", "message": f"Custom sign '{sign_name}' loaded successfully!"}), 200


def generate_custom_sign_detection():
    """ Generate video stream for detecting custom signs. """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame and extract keypoints for LSTM
        frame_resized = cv2.resize(frame, (640, 480))
        keypoints = preprocess_for_lstm(frame_resized)

        if keypoints is not None and current_custom_sign_name:
            # Load the stored keypoints for the custom sign
            custom_sign_keypoints = load_custom_sign(current_custom_sign_name)

            if custom_sign_keypoints is not None:
                # Detect the custom sign
                is_detected = detect_custom_sign(lstm_model, keypoints, custom_sign_keypoints)
                if is_detected:
                    predicted_sign = f"Custom sign '{current_custom_sign_name}' detected!"
                else:
                    predicted_sign = "Custom sign not detected"
            else:
                predicted_sign = "Error loading custom sign keypoints"
        else:
            predicted_sign = "No sign detected"

        # Display predicted custom sign on the frame
        cv2.putText(frame, predicted_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/custom_sign_feed')
def custom_sign_feed():
    """ Route for custom sign video streaming. """
    return Response(generate_custom_sign_detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
