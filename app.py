from flask import Flask, request
import cv2
import mediapipe as mp
import tempfile
import os

app = Flask(__name__)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        file.save(temp.name)
        temp_path = temp.name  # Store the path for later use

    image = cv2.imread(temp_path)
    pose = mp_pose.Pose()
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    handres = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    hlm = handres.multi_hand_landmarks    

    label = "none"
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        #print("Landmarks:",lm)  # Debugging to check what landmarks are detected
        
        # Head detection (using NOSE)
        if lm[mp_pose.PoseLandmark.NOSE].visibility > 0.7:
            label = "head"      
        
        elif lm[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.7 or \
             lm[mp_pose.PoseLandmark.LEFT_KNEE].visibility > 0.7 or \
             lm[mp_pose.PoseLandmark.LEFT_ANKLE].visibility > 0.7:
            label = "leg"
    
    if hlm:
        label = "hand"

    os.unlink(temp_path)  # Now safe to delete
    return label



@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
