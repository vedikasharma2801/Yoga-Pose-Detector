import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model and Encoder ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'yoga_pose_model.keras')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

try:
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    # Exit or handle the error appropriately if the app can't run without them
    exit()

# --- Initialize MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    """
    Generator function to capture video from webcam, process it, and yield frames.
    """
    cap = cv2.VideoCapture(0) # Use 0 for the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # --- Pose Detection and Prediction ---
            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # --- Extract Keypoints and Predict ---
            try:
                if results.pose_landmarks:
                    # Extract landmarks
                    landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
                    landmarks = np.expand_dims(landmarks, axis=0) # Add batch dimension
                    landmarks = np.expand_dims(landmarks, axis=1) # Add timestep dimension for LSTM

                    # Make prediction
                    prediction = model.predict(landmarks)
                    predicted_class_index = np.argmax(prediction)
                    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
                    prediction_accuracy = prediction[0][predicted_class_index]

                    # --- Display Output on Frame ---
                    # Display predicted asana
                    cv2.putText(frame, f"Asana: {predicted_class_label}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # Display accuracy
                    cv2.putText(frame, f"Accuracy: {prediction_accuracy:.2f}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw the landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            except Exception as e:
                # print(f"Error during prediction or drawing: {e}")
                pass # Continue to the next frame

            # --- Encode frame to JPEG ---
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in the correct format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)