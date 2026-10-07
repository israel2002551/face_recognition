import face_recognition
import cv2
import numpy as np
import time
import pickle
import os
import base64
import threading
import json
import paho.mqtt.client as mqtt
from imutils import paths
from picamera2 import Picamera2
from gpiozero import LED
from flask import Flask, request, jsonify, render_template_string

# --- CONFIGURATION ---
output = LED(14) # Actual GPIO pin control for the door lock
MQTT_BROKER = "test.mosquitto.org" 
MQTT_PORT = 1883
TOPIC_REGISTER = "company/face/register"
TOPIC_APPROVED = "pi/face/approved"

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pickle"
AUTHORIZED_NAMES = ["john", "alice", "bob"]

# --- GLOBALS & LOCKS ---
known_face_encodings = []
known_face_names = []
model_lock = threading.Lock() 
app = Flask(name)

# --- CORE FUNCTIONS ---
def load_encodings():
    global known_face_encodings, known_face_names
    print("[INFO] Loading encodings...")
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.loads(f.read())
        with model_lock:
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    else:
        print("[WARNING] No encodings found. Please register a user.")

def train_model():
    print("[INFO] Start processing faces...")
    imagePaths = list(paths.list_images(DATASET_DIR))
    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    print("[INFO] Serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(ENCODINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))
    
    load_encodings()
    print("[INFO] Training complete and live model updated.")

# --- MQTT SETUP ---
def on_mqtt_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}")
    client.subscribe(TOPIC_APPROVED)

def on_mqtt_message(client, userdata, msg):
    print(f"[MQTT] Message received on topic {msg.topic}")
    try:
        data = json.loads(msg.payload.decode())
        name = data.get("name")
        image_data_url = data.get("image")
        
        print(f"[MQTT] Approval received for: {name}. Saving and training...")
        
        header, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        user_dir = os.path.join(DATASET_DIR, name)
        os.makedirs(user_dir, exist_ok=True)
        file_path = os.path.join(user_dir, f"{name}_{int(time.time())}.jpg")
        
        with open(file_path, "wb") as f:
            f.write(image_bytes)
            
        if name not in AUTHORIZED_NAMES:
            AUTHORIZED_NAMES.append(name)
            
        threading.Thread(target=train_model).start()
        
    except Exception as e:
        print(f"[MQTT ERROR] Failed to process incoming approval: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# --- FLASK WEB ROUTES (Professional Enterprise UI) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Identity Registration Portal</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
[07/10/2026 11:50] Efe Israel: :root {
            --bg-body: #f0f4f8;
            --bg-card: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #64748b;
            --accent-color: #0f172a; 
            --accent-hover: #334155;
            --border-color: #e2e8f0;
            --focus-ring: rgba(15, 23, 42, 0.15);
            --success-bg: #f0fdf4;
            --success-border: #bbf7d0;
            --success-text: #166534;
            --error-bg: #fef2f2;
            --error-border: #fecaca;
            --error-text: #991b1b;
        }

        * { box-sizing: border-box; }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
            color: var(--text-primary);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }

        .portal-card {
            background: var(--bg-card);
            width: 100%;
            max-width: 480px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.5);
            overflow: hidden;
        }

        .card-header {
            padding: 32px 32px 24px;
            text-align: center;
        }

        .logo-placeholder {
            width: 48px;
            height: 48px;
            background: var(--accent-color);
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .card-header h2 {
            margin: 0;
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.025em;
        }

        .card-header p {
            margin: 8px 0 0 0;
            font-size: 14px;
            color: var(--text-secondary);
        }

        .card-body {
            padding: 0 32px 32px;
        }

        .form-group {
            margin-bottom: 24px;
            position: relative;
        }

        label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-wrapper {
            position: relative;
        }

        .input-icon {
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: #94a3b8;
            width: 18px;
            height: 18px;
        }

        input[type="text"] {
            width: 100%;
            padding: 12px 12px 12px 40px;
            font-size: 15px;
            background-color: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            transition: all 0.2s ease;
            outline: none;
            color: var(--text-primary);
            font-weight: 500;
        }

        input[type="text"]:focus {
            background-color: #ffffff;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 4px var(--focus-ring);
        }

        .video-container {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 24px;
            background: #000;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
        }

        video {
            width: 100%;
            height: 320px;
            object-fit: cover;
            display: block;
            transform: scaleX(-1);
        }
[07/10/2026 11:50] Efe Israel: .scan-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%);
            background-size: 100% 4px;
            z-index: 2;
        }

        .reticle {
            position: absolute;
            width: 40px; height: 40px;
            border-color: rgba(255,255,255,0.7);
            border-style: solid;
            z-index: 3;
        }
        .reticle.top-left { top: 20px; left: 20px; border-width: 3px 0 0 3px; border-top-left-radius: 8px; }
        .reticle.top-right { top: 20px; right: 20px; border-width: 3px 3px 0 0; border-top-right-radius: 8px; }
        .reticle.bottom-left { bottom: 20px; left: 20px; border-width: 0 0 3px 3px; border-bottom-left-radius: 8px; }
        .reticle.bottom-right { bottom: 20px; right: 20px; border-width: 0 3px 3px 0; border-bottom-right-radius: 8px; }

        .live-indicator {
            position: absolute;
            top: 16px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.6);
            color: #fff;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
            backdrop-filter: blur(4px);
            z-index: 10;
            letter-spacing: 0.05em;
        }

        .live-dot {
            width: 6px;
            height: 6px;
            background-color: #ef4444;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        .btn-primary {
            width: 100%;
            background-color: var(--accent-color);
            color: white;
            border: none;
            padding: 14px;
            font-size: 15px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }

        .btn-primary:hover {
            background-color: var(--accent-hover);
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .btn-primary:active {
            transform: translateY(0);
        }

        .btn-primary:disabled {
            background-color: #cbd5e1;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        #status {
            margin-top: 20px;
            padding: 14px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            display: none;
            align-items: flex-start;
            gap: 10px;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .status-info { background-color: #f8fafc; color: #334155; border: 1px solid #e2e8f0; display: flex !important; }
        .status-success { background-color: var(--success-bg); color: var(--success-text); border: 1px solid var(--success-border); display: flex !important; }
        .status-error { background-color: var(--error-bg); color: var(--error-text); border: 1px solid var(--error-border); display: flex !important; }

        canvas { display: none; }
    </style>
</head>
<body>
[07/10/2026 11:50] Efe Israel: <div class="portal-card">
        <div class="card-header">
            <div class="logo-placeholder">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                </svg>
            </div>
            <h2>Biometric Registration</h2>
            <p>Secure Identity Verification System</p>
        </div>

        <div class="card-body">
            <div class="form-group">
                <label for="username">Employee ID / Full Name</label>
                <div class="input-wrapper">
                    <svg class="input-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                    </svg>
                    <input type="text" id="username" placeholder="e.g. John Doe" required autocomplete="off">
                </div>
            </div>

            <div class="video-container">
                <div class="scan-overlay"></div>
                <div class="reticle top-left"></div>
                <div class="reticle top-right"></div>
                <div class="reticle bottom-left"></div>
                <div class="reticle bottom-right"></div>
                
                <div class="live-indicator">
                    <div class="live-dot"></div> LIVE
                </div>
                <video id="video" autoplay playsinline muted></video>
            </div>

            <button id="captureBtn" class="btn-primary">
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path>
                </svg>
                Capture & Submit Request
            </button>
            
            <div id="status"></div>
            <canvas id="canvas" width="1280" height="720"></canvas>
        </div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const captureBtn = document.getElementById('captureBtn');
        const statusText = document.getElementById('status');

        navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" } })
            .then(stream => { 
                video.srcObject = stream; 
            })
            .catch(err => { 
                console.error("Error accessing webcam: ", err); 
                statusText.className = 'status-error';
                statusText.innerHTML = <strong>Error:</strong> Camera access denied. Please check your browser permissions.;
            });

        captureBtn.addEventListener('click', () => {
            const username = document.getElementById('username').value.trim();
            
            if (!username) { 
                statusText.className = 'status-error';
                statusText.innerHTML = <strong>Attention:</strong> Please enter a valid Employee ID or Name.;
                document.getElementById('username').focus();
                return; 
            }

            captureBtn.disabled = true;
            captureBtn.innerHTML = `
                <svg class="animate-spin" width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="animation: spin 1s linear infinite;">
[07/10/2026 11:50] Efe Israel: <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                </svg>
                Processing...
            ;
            
            statusText.className = 'status-info';
            statusText.innerHTML = <strong>Processing:</strong> Encrypting biometric data and contacting HQ...;

            if (!document.getElementById('spinner-style')) {
                const style = document.createElement('style');
                style.id = 'spinner-style';
                style.innerHTML = @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } };
                document.head.appendChild(style);
            }

            const context = canvas.getContext('2d');
            context.translate(canvas.width, 0);
            context.scale(-1, 1);
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.85); 
            
            fetch('/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: username, image: imageData })
            })
            .then(response => {
                if (!response.ok) throw new Error('Network response was not ok');
                return response.json();
            })
            .then(data => {
                statusText.className = 'status-success';
                statusText.innerHTML = <strong>Success:</strong> ${data.message};
                document.getElementById('username').value = '';
            })
            .catch(err => {
                console.error(err);
                statusText.className = 'status-error';
                statusText.innerHTML = <strong>System Error:</strong> Failed to connect to secure server.;
            })
            .finally(() => {
                setTimeout(() => {
                    captureBtn.disabled = false;
                    captureBtn.innerHTML = 
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
                            <path stroke-linecap="round" stroke-linejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path>
                        </svg>
                        Capture & Submit Request
                    `;
                }, 1500);
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    image_data_url = data.get('image')

    if not name or not image_data_url:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    payload = json.dumps({"name": name, "image": image_data_url})
    mqtt_client.publish(TOPIC_REGISTER, payload)
    
    print(f"[WEB] Published registration request for {name} to MQTT.")
    return jsonify({"status": "pending", "message": "Request sent! Awaiting HQ approval."}), 200

def start_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# --- RECOGNITION LOOP (PiCamera Version) ---
def main_recognition_loop():
    load_encodings()
    
    # Initialize Pi Camera 2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
    picam2.start()

    cv_scaler = 4
    frame_count = 0
    start_time = time.time()
    fps = 0

    try:
        while True:
            # Capture frame from the Raspberry Pi camera
            frame = picam2.capture_array()
            resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
            rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_resized_frame)
            face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
            
            face_names = []
            authorized_face_detected = False
            
            with model_lock:
                local_known_encodings = list(known_face_encodings)
                local_known_names = list(known_face_names)

            for face_encoding in face_encodings:
                if len(local_known_encodings) > 0:
                    matches = face_recognition.compare_faces(local_known_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(local_known_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = local_known_names[best_match_index]
                            if name in AUTHORIZED_NAMES:
                                authorized_face_detected = True
                    face_names.append(name)
                else:
                    face_names.append("Unknown")

            # Control the physical GPIO pin based on detection
            if authorized_face_detected:
                output.on()
            else:
                output.off()

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= cv_scaler
                right *= cv_scaler
                bottom *= cv_scaler
                left *= cv_scaler
                
                color = (0, 255, 0) if name in AUTHORIZED_NAMES else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.rectangle(frame, (left -3, top - 35), (right+3, top), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video', frame)
            
            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        # Close Pi camera and clear GPIO
        picam2.stop()
        cv2.destroyAllWindows()
        output.off()

if name == "main":
    # Start the Flask web app for registration on a background thread
    web_thread = threading.Thread(target=start_flask, daemon=True)
    web_thread.start()
    
    # Run the live face recognition feed simultaneously
    main_recognition_loop()
