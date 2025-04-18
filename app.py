from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
from PIL import Image
import csv
import mediapipe as mp

from model import KeyPointClassifier

# === Initialize Flask App ===
app = Flask(__name__)
CORS(app)

# === Load MediaPipe and Gesture Model ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
keypoint_classifier = KeyPointClassifier()

# === Load Label Map ===
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

# === Helper Functions ===
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp = landmark_list.copy()
    base_x, base_y = temp[0]
    
    # Convert to relative coordinates
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    
    # Flatten and normalize
    temp = np.array(temp).flatten()
    max_val = np.max(np.abs(temp)) if np.max(np.abs(temp)) != 0 else 1
    return (temp / max_val).tolist()

# === Routes ===
@app.route('/')
def home():
    return "👋 Flask gesture recognition server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400

        # Decode base64 image
        image_b64 = data['image'].split(',')[1]
        image_data = base64.b64decode(image_b64)
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image_pil)

        # Run hand detection
        results = hands.process(image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark_list = calc_landmark_list(image, hand_landmarks)
            processed = pre_process_landmark(landmark_list)
            gesture_id = keypoint_classifier(processed)
            gesture = keypoint_classifier_labels[gesture_id]
            return jsonify({'gesture': gesture})

        return jsonify({'gesture': 'No hand detected'})

    except Exception as e:
        print("❌ Error in /predict:", e)
        return jsonify({'error': str(e)}), 500

# === Entry Point ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
