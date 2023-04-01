from flask import Flask, render_template, jsonify, request, send_from_directory
from fileinput import filename
import cv2
import tempfile
from pathlib import Path
import mediapipe as mp
import time
import os
import numpy as np
import math

app = Flask(__name__, template_folder='.')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(l1, l2):
    return np.arccos((l1[0]*l2[0] + l1[1]*l2[1])/(math.sqrt(l1[0]**2+l1[1]**2)*math.sqrt(l2[0]**2+l2[1]**2)))

def calculate_angles(positions):
    angles = []
    for frame in range(len(positions)):
        frame_positions = positions[frame]
        # Left arm vector
        left_arm_vector_1 = [frame_positions[0] - frame_positions[2], frame_positions[1] - frame_positions[3]]
        left_arm_vector_2 = [frame_positions[4] - frame_positions[2], frame_positions[5] - frame_positions[3]]
        left_arm_angle = calculate_angle(left_arm_vector_1, left_arm_vector_2)
        # Left hand vector
        left_hand_vector_1 = [frame_positions[2] - frame_positions[4], frame_positions[3] - frame_positions[5]]
        left_hand_vector_2 = [frame_positions[6] - frame_positions[4], frame_positions[7] - frame_positions[5]]
        left_hand_angle = calculate_angle(left_hand_vector_1, left_hand_vector_2)
        # Left leg vector
        left_leg_vector_1 = [frame_positions[8] - frame_positions[10], frame_positions[9] - frame_positions[11]]
        left_leg_vector_2 = [frame_positions[12] - frame_positions[10], frame_positions[13] - frame_positions[11]]
        left_leg_angle = calculate_angle(left_leg_vector_1, left_leg_vector_2)
        # Left foot vector
        left_foot_vector_1 = [frame_positions[10] - frame_positions[12], frame_positions[11] - frame_positions[13]]
        left_foot_vector_2 = [frame_positions[14] - frame_positions[12], frame_positions[15] - frame_positions[13]]
        left_foot_angle = calculate_angle(left_foot_vector_1, left_foot_vector_2)
        
        # Right arm vector
        right_arm_vector_1 = [frame_positions[16] - frame_positions[18], frame_positions[17] - frame_positions[19]]
        right_arm_vector_2 = [frame_positions[20] - frame_positions[18], frame_positions[21] - frame_positions[19]]
        right_arm_angle = calculate_angle(right_arm_vector_1, right_arm_vector_2)
        # Right hand vector
        right_hand_vector_1 = [frame_positions[18] - frame_positions[20], frame_positions[19] - frame_positions[21]]
        right_hand_vector_2 = [frame_positions[22] - frame_positions[20], frame_positions[23] - frame_positions[21]]
        right_hand_angle = calculate_angle(right_hand_vector_1, right_hand_vector_2)
        # Right leg vector
        right_leg_vector_1 = [frame_positions[24] - frame_positions[26], frame_positions[25] - frame_positions[27]]
        right_leg_vector_2 = [frame_positions[28] - frame_positions[26], frame_positions[29] - frame_positions[27]]
        right_leg_angle = calculate_angle(right_leg_vector_1, right_leg_vector_2)
        # Right foot vector
        right_foot_vector_1 = [frame_positions[26] - frame_positions[28], frame_positions[27] - frame_positions[29]]
        right_foot_vector_2 = [frame_positions[30] - frame_positions[28], frame_positions[31] - frame_positions[29]]
        right_foot_angle = calculate_angle(right_foot_vector_1, right_foot_vector_2)

        angles.append([left_arm_angle, left_hand_angle, left_leg_angle, left_foot_angle, right_arm_angle, right_hand_angle, right_leg_angle, right_foot_angle])
    return angles

def calculate_score(a1, a2):
    score = 0
    for frame in range(len(a1)):
        a1_angles = np.array(a1[frame])
        a2_angles = np.array(a2[frame])
        angle_diff = a1_angles - a2_angles
        score += np.linalg.norm(angle_diff)
    score /= len(a1) * -1
    return np.exp(score)

@app.route('/')  
def main():  
    return render_template("templates/index.html")  
  
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        # Read in video file and store temporarily
        uploaded_file = request.files['file']
        with tempfile.TemporaryDirectory() as td:
            temp_filename = Path(td) / 'uploaded_video'
            uploaded_file.save(temp_filename)
            cap = cv2.VideoCapture(str(temp_filename))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (width, height)
            frames_uploaded = []
            # Capture frames.
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                # Get fps.
                fps = cap.get(cv2.CAP_PROP_FPS)
                # Get height and width of the frame.
                h, w = image.shape[:2]
                # Convert the BGR image to RGB.
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Process the image.
                keypoints = pose.process(image)
                # Convert the image back to BGR.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Use lm and lmPose as representative of the following methods.
                lm = keypoints.pose_landmarks
                lmPose = mp_pose.PoseLandmark
                if lm is None:
                    continue
                # Acquire the landmark coordinates.
                # Once aligned properly, left or right should not be a concern.      
                # Left shoulder.
                l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                # Left elbow.
                l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
                l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
                # Left wrist.
                l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
                l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
                # Left index finger
                l_index_x = int(lm.landmark[lmPose.LEFT_INDEX].x * w)
                l_index_y = int(lm.landmark[lmPose.LEFT_INDEX].y * h)
                # Left hip
                l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
                l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
                # Left knee
                l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
                l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
                # Left ankle
                l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
                l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
                # Left foot index
                l_foot_x = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].x * w)
                l_foot_y = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].y * h)
                # Right side body parts
                # Right shoulder.
                r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                # Right elbow.
                r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
                r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
                # Right wrist.
                r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
                r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
                # Right index finger
                r_index_x = int(lm.landmark[lmPose.RIGHT_INDEX].x * w)
                r_index_y = int(lm.landmark[lmPose.RIGHT_INDEX].y * h)
                # Right hip
                r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
                r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
                # Right knee
                r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
                r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
                # Right ankle
                r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
                r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)
                # Right foot index
                r_foot_x = int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x * w)
                r_foot_y = int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].y * h)
                frames_uploaded.append([l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y, l_index_x, l_index_y, l_hip_x, l_hip_y, l_knee_x, l_knee_y, l_ankle_x, l_ankle_y, l_foot_x, l_foot_y,
                              r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y, r_wrist_x, r_wrist_y, r_index_x, r_index_y, r_hip_x, r_hip_y, r_knee_x, r_knee_y, r_ankle_x, r_ankle_y, r_foot_x, r_foot_y])
            cap.release()
        # Read in video file and store temporarily
        baseline_file = "IMG_1940.MOV"
        cap = cv2.VideoCapture(str(baseline_file))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        frames_baseline = []
        # Capture frames.
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Null.Frames")
                break
            # Get fps.
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Get height and width of the frame.
            h, w = image.shape[:2]
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process the image.
            keypoints = pose.process(image)
            # Convert the image back to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Use lm and lmPose as representative of the following methods.
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark
            if lm is None:
                continue
            # Acquire the landmark coordinates.
            # Once aligned properly, left or right should not be a concern.      
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            # Left elbow.
            l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
            l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
            # Left wrist.
            l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
            l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
            # Left index finger
            l_index_x = int(lm.landmark[lmPose.LEFT_INDEX].x * w)
            l_index_y = int(lm.landmark[lmPose.LEFT_INDEX].y * h)
            # Left hip
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
            # Left knee
            l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
            l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
            # Left ankle
            l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
            l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
            # Left foot index
            l_foot_x = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].x * w)
            l_foot_y = int(lm.landmark[lmPose.LEFT_FOOT_INDEX].y * h)
            # Right side body parts
            # Right shoulder.
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            # Right elbow.
            r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
            r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)
            # Right wrist.
            r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
            r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)
            # Right index finger
            r_index_x = int(lm.landmark[lmPose.RIGHT_INDEX].x * w)
            r_index_y = int(lm.landmark[lmPose.RIGHT_INDEX].y * h)
            # Right hip
            r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
            r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
            # Right knee
            r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
            r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)
            # Right ankle
            r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
            r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)
            # Right foot index
            r_foot_x = int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x * w)
            r_foot_y = int(lm.landmark[lmPose.RIGHT_FOOT_INDEX].y * h)
            frames_baseline.append([l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y, l_index_x, l_index_y, l_hip_x, l_hip_y, l_knee_x, l_knee_y, l_ankle_x, l_ankle_y, l_foot_x, l_foot_y,
                            r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y, r_wrist_x, r_wrist_y, r_index_x, r_index_y, r_hip_x, r_hip_y, r_knee_x, r_knee_y, r_ankle_x, r_ankle_y, r_foot_x, r_foot_y])
        cap.release()
        angles_baseline = calculate_angles(frames_baseline)
        angles_uploaded = calculate_angles(frames_uploaded)
        min_len = min(len(angles_baseline), len(angles_uploaded))
        score = calculate_score(angles_baseline[0:min_len], angles_uploaded[0:min_len])
        print("Score:", score)
    return render_template("templates/acknowledgement.html", name = uploaded_file.filename)  