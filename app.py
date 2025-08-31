from flask import Flask, render_template, Response, request, jsonify, redirect, session, url_for, flash
import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import threading
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import re
import time

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "supersecretkey"
db = SQLAlchemy(app)

# User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(30), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def check_password(self, password):
        return bcrypt.checkpw(password.encode("utf-8"), self.password.encode("utf-8"))

# Deepfake Detection Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the pretrained MobileNet model
model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=2)
model.load_state_dict(torch.load('mobilenet_model.pth', map_location=device))
model.to(device)
model.eval()

# Video processing variables
video_source = 0  # Default to webcam
video_lock = threading.Lock()
processing = False
cap = None

# Prediction counters
fake_count = 0
real_count = 0
total_frames = 0

# Authentication Routes
@app.route("/")
def index():
    return redirect(url_for('login'))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not all([username, email, password, confirm_password]):
            flash("All fields are required!", "error")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords don't match!", "error")
            return redirect(url_for('register'))

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email format!", "error")
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered!", "error")
            return redirect(url_for('register'))

        new_user = User(username, email, password)
        db.session.add(new_user)
        db.session.commit()

        session['user'] = username
        flash("Registration successful!", "success")
        return redirect(url_for('home'))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user'] = user.username
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password.", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out", "info")
    return redirect(url_for('login'))

# Application Pages
@app.route("/home")
def home():
    if "user" not in session:
        flash("Please login first", "error")
        return redirect(url_for('login'))
    return render_template("home.html", username=session['user'])

@app.route("/detection")
def detection():
    if "user" not in session:
        flash("Please login first", "error")
        return redirect(url_for('login'))
    return render_template("detection.html", username=session['user'])

@app.route("/pricing")
def pricing():
    return render_template("pricing.html", username=session.get('user'))

@app.route("/contact")
def contact():
    return render_template("contact.html", username=session.get('user'))

# Deepfake Detection Functions
def predict_frame(frame):
    global fake_count, real_count, total_frames
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        total_frames += 1
        if predicted.item() == 1:
            real_count += 1
            return "Real"
        else:
            fake_count += 1
            return "Fake"
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error"

def generate_frames():
    global video_source, cap, processing, fake_count, real_count, total_frames
    with video_lock:
        if isinstance(video_source, str) and video_source.endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = cv2.VideoCapture(video_source)

    processing = True
    while processing and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        label = predict_frame(frame)

        # Add label to the frame
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if label == "Real" else (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    processing = False
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_video_source', methods=['POST'])
def set_video_source():
    global video_source, real_count, fake_count, total_frames, cap
    
    # Reset counters
    real_count = 0
    fake_count = 0
    total_frames = 0

    if 'file' not in request.files:
        with video_lock:
            video_source = 0
        return jsonify({"message": "Switched to webcam"})

    file = request.files['file']
    if file.filename == '':
        with video_lock:
            video_source = 0
        return jsonify({"message": "No file selected"})

    if file:
        # Create static directory if it doesn't exist
        os.makedirs('static/uploads', exist_ok=True)
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        
        with video_lock:
            video_source = file_path
        
        return jsonify({"message": "Video uploaded and processing started"})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global processing, cap, real_count, fake_count, total_frames
    processing = False
    if cap:
        cap.release()
    real_count = 0
    fake_count = 0
    total_frames = 0
    return jsonify({"message": "Video stopped"})

@app.route('/prediction_stats')
def prediction_stats():
    global real_count, fake_count, total_frames
    if total_frames == 0:
        return jsonify({"real_percent": 0, "fake_percent": 0})
    
    real_percent = (real_count / total_frames) * 100
    fake_percent = (fake_count / total_frames) * 100
    return jsonify({
        "real_percent": round(real_percent, 2),
        "fake_percent": round(fake_percent, 2),
        "total_frames": total_frames
    })

@app.route('/final_result')
def final_result():
    global real_count, fake_count, total_frames
    if total_frames == 0:
        return jsonify({"result": "No video processed"})
    
    if real_count > fake_count:
        return jsonify({"result": "Real", "Percentage": round((real_count/total_frames)*100, 2)})
    elif fake_count > real_count:
        return jsonify({"result": "Fake", "Percentage": round((fake_count/total_frames)*100, 2)})
    else:
        return jsonify({"result": "Uncertain", "Percentage": 50})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True, threaded=True)