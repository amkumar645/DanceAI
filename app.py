from flask import Flask, render_template, request, redirect, session
import cv2
import tempfile
from pathlib import Path
import mediapipe as mp
import numpy as np
import math
import db
import json
from vidgear.gears import CamGear

app = Flask(__name__, template_folder='.')
app.secret_key = "cat"
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_normalized(positions):
    for frame in range(len(positions)):
        chest_x = positions[frame][-2]
        chest_y = positions[frame][-1]
        for position in range(len(positions[frame])):
            if position % 2 == 0:
                positions[frame][position] -= chest_x
            else:
                positions[frame][position] -= chest_y
        return positions

def calculate_normalized_score(n1, n2, offset):
    score = 0
    worst_frame_score = 0
    worst_frame = -1
    total_count = 0
    for frame in range(len(n1)):
        n1_score = np.array(n1[frame])
        n2_score = np.array(n2[frame])
        n_diff = n1_score - n2_score
        frame_score = np.linalg.norm(n_diff)
        if frame_score * 0.01 > worst_frame_score:
            worst_frame_score = frame_score * 0.001
            worst_frame = frame
        if not np.isnan(frame_score):
            score += frame_score * 0.001
            total_count += 1
    score /= total_count * -1
    current_frame = (worst_frame + offset) / 30
    frame_1 = math.floor(current_frame)
    frame_2 = math.ceil(current_frame)
    # print("Check your poses between seconds", frame_1, "and", frame_2)
    return np.exp(score)

def get_score(uploaded_file, youtube_url):
    with tempfile.TemporaryDirectory() as td:
        temp_filename = Path(td) / 'uploaded_video'
        uploaded_file.save(temp_filename)
        cap = cv2.VideoCapture(str(temp_filename))
        frames_uploaded = []
        # Capture frames.
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
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
            # Calculate chest for normalizing
            chest_x = (l_shldr_x + r_shldr_x + l_hip_x + r_hip_x)/4
            chest_y = (l_shldr_y + r_shldr_y + l_hip_y + r_hip_y)/4
            frames_uploaded.append([l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y, l_index_x, l_index_y, l_hip_x, l_hip_y, l_knee_x, l_knee_y, l_ankle_x, l_ankle_y, l_foot_x, l_foot_y,
                            r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y, r_wrist_x, r_wrist_y, r_index_x, r_index_y, r_hip_x, r_hip_y, r_knee_x, r_knee_y, r_ankle_x, r_ankle_y, r_foot_x, r_foot_y,
                            chest_x, chest_y])
        cap.release()
    # Baseline
    stream = CamGear(source=youtube_url, stream_mode = True, logging=False).start() # YouTube Video URL as input
    frames_baseline = []
    while True:
        image = stream.read()
        if image is None:
            break
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
        # Calculate chest for normalizing
        chest_x = (l_shldr_x + r_shldr_x + l_hip_x + r_hip_x)/4
        chest_y = (l_shldr_y + r_shldr_y + l_hip_y + r_hip_y)/4
        frames_baseline.append([l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y, l_index_x, l_index_y, l_hip_x, l_hip_y, l_knee_x, l_knee_y, l_ankle_x, l_ankle_y, l_foot_x, l_foot_y,
                        r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y, r_wrist_x, r_wrist_y, r_index_x, r_index_y, r_hip_x, r_hip_y, r_knee_x, r_knee_y, r_ankle_x, r_ankle_y, r_foot_x, r_foot_y,
                        chest_x, chest_y])
    stream.stop()
    normalized_baseline = calculate_normalized(frames_baseline)
    normalized_uploaded = calculate_normalized(frames_uploaded)
    len_baseline = len(normalized_baseline)
    len_uploaded = len(normalized_uploaded)
    max_score = 0
    if len_baseline < len_uploaded:
        offset = len_uploaded - len_baseline
        for i in range(offset + 1):
            score = calculate_normalized_score(normalized_baseline[0:len_baseline], normalized_uploaded[i:len_baseline + i], 0)
            if score > max_score:
                max_score = score
    else:
        offset = len_baseline - len_uploaded
        for i in range(offset + 1):
            score = calculate_normalized_score(normalized_baseline[i:len_uploaded + i], normalized_uploaded[0:len_uploaded], i)
            if score > max_score:
                max_score = score
    return max_score

# Page routes begin here
# Home page
@app.route('/')
def home():
    return render_template("templates/home.html")  
# Login route
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        data = json.loads(request.form.to_dict()['event_data'])
        user = db.user_collection.find_one({'username': data['username']})
        if user is None:
            db.user_collection.insert_one({
                "username": data['username'],
                "password": data['password'],
                "kpop_beginner": 0,
                "kpop_intermediate": 0,
                "kpop_advanced": 0
            })
        session['username'] = data['username']
    return "/learn"
# Learn page
@app.route('/learn')
def learn_home():
    username = session['username']
    user = db.user_collection.find_one({'username': username})
    if user is None:
        return redirect("/")
    return render_template("templates/learn.html", username=username)  

@app.route('/learn/kpop')  
def learn_kpop():
    username = session['username']
    user = db.user_collection.find_one({'username': username})
    if user is None:
        return redirect("/")    
    return render_template("templates/learnkpop.html", 
                            username=username,
                            kpop_beginner = user['kpop_beginner'],
                            kpop_intermediate = user['kpop_intermediate'],
                            kpop_advanced = user['kpop_advanced'])  

@app.route('/learn/kpop/beginner', methods=['POST'])  
def learn_kpop_beginner():
    if request.method == 'POST':  
        # Read in video file and store temporarily
        uploaded_file = request.files['file']
        score = get_score(uploaded_file, "https://youtube.com/shorts/Md9huDZcKl8?feature=share")
        username = session['username']
        filter = {"username": username}
        new_vals = { "$set":
            {
                "username": username,
                "kpop_beginner": score
            }
        }
        db.user_collection.update_one(filter, new_vals)
        return redirect('/learn/kpop') 
    
@app.route('/learn/kpop/intermediate', methods=['POST'])  
def learn_kpop_intermediate():
    if request.method == 'POST':  
        # Read in video file and store temporarily
        uploaded_file = request.files['file']
        score = get_score(uploaded_file, "https://youtube.com/shorts/uhFGlDWah10?feature=share")
        username = session['username']
        filter = {"username": username}
        new_vals = { "$set":
            {
                "username": username,
                "kpop_intermediate": score
            }
        }
        db.user_collection.update_one(filter, new_vals)
        return redirect('/learn/kpop') 

@app.route('/learn/kpop/advanced', methods=['POST'])  
def learn_kpop_advanced():
    if request.method == 'POST':  
        # Read in video file and store temporarily
        uploaded_file = request.files['file']
        score = get_score(uploaded_file, "https://youtube.com/shorts/I-pY4jxmA-k?feature=share")
        username = session['username']
        filter = {"username": username}
        new_vals = { "$set":
            {
                "username": username,
                "kpop_advanced": score
            }
        }
        db.user_collection.update_one(filter, new_vals)
        return redirect('/learn/kpop') 

# Explore page
@app.route('/explore')
def explore_home():
    explores = db.explore_collection.find()
    return render_template("templates/explore.html", explores=explores)  

@app.route('/explore/add', methods=['POST'])
def explore_add():
    if request.method == 'POST':
        data = json.loads(request.form.to_dict()['event_data'])
        username = session['username']
        url = data['embedded']
        id = url.split("v=")[1]
        embedded = "https://www.youtube.com/embed/" + id
        db.explore_collection.insert_one({
            "title": data['title'],
            "embedded": embedded,
            "type": data['type'],
            "creator": username,
            "leaderboard_users": ["jiaweim", "amkumar"],
            "leaderboard_scores": [0.85, 0.47]
        })
    return "SUCCESS"

@app.route('/explore/leaderboard', methods=['GET'])
def explore_leaderboard():
    embedded = request.args.get('query')
    explore = db.explore_collection.find_one({'embedded': embedded})
    users = explore['leaderboard_users']
    scores = explore['leaderboard_scores']
    leaderboard_string = ''
    for i in range(len(users)):
        leaderboard_string += users[i] + ": " + str(scores[i]) + "<br>"
    return leaderboard_string

# Explore page
@app.route('/connect')
def connect_home():
    return render_template("templates/connect.html")  