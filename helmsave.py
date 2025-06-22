import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import threading
from ultralytics import YOLO
import customtkinter as ctk
from PIL import Image, ImageTk
import datetime
import os

# Inisialisasi YOLOv8
helmet_model = YOLO("best3.pt")

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# Load known faces
known_faces = []
known_names = []

def load_and_encode_image(image_path, name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img)
    if encodings:
        known_faces.append(encodings[0])
        known_names.append(name)
    else:
        print(f"No face detected in {name}")

# Load wajah
load_and_encode_image('image/al_depan.jpg', 'Al')
load_and_encode_image('image/ilham_depan.jpg', 'Ilham')
load_and_encode_image('image/wahyu.jpg', 'Wahyu')
load_and_encode_image('image/irpan.jpg', 'Irvan')
load_and_encode_image('image/erika.jpg', 'Erika')

# Inisialisasi Streaming Kamera
#URL = 'http://10.12.12.246:8080/?action=stream'
#cap = cv2.VideoCapture(URL)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka kamera.")
    exit()

frame = None
lock = threading.Lock()

def capture_frames():
    global frame
    while True:
        ret, new_frame = cap.read()
        if not ret:
            print("Gagal mengambil frame")
            break
        with lock:
            frame = new_frame.copy()

threading.Thread(target=capture_frames, daemon=True).start()

# Fungsi IOU dan cek helm
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def is_helmet_above_face(face_box, helmet_box):
    fx1, fy1, fx2, fy2 = face_box
    hx1, hy1, hx2, hy2 = helmet_box
    vertical_check = hy2 <= fy2 and hy1 < fy1 and abs(fy1 - hy2) < (fy2 - fy1) * 0.6
    horizontal_overlap = hx1 < fx2 and hx2 > fx1
    return vertical_check and horizontal_overlap

# === GUI dengan CustomTkinter ===
ctk.set_appearance_mode("dark")  # Mode Gelap
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Helmet and Face Recognition System")
app.geometry("1500x850")
app.resizable(False, False)

# Main Frame
main_frame = ctk.CTkFrame(app, corner_radius=20, fg_color="black")
main_frame.pack(padx=20, pady=20, fill="both", expand=True)

# Title
title_label = ctk.CTkLabel(main_frame, text="Helmet and Face Recognition", font=("Arial", 36, "bold"), text_color="white")
title_label.pack(pady=(20, 20))

# Content Frame
content_frame = ctk.CTkFrame(main_frame, corner_radius=20, fg_color="black")
content_frame.pack(padx=10, pady=10, fill="both", expand=True)

content_frame.grid_columnconfigure(0, weight=3)
content_frame.grid_columnconfigure(1, weight=1)
content_frame.grid_rowconfigure(0, weight=1)

# Kamera Frame
camera_frame = ctk.CTkFrame(content_frame, corner_radius=20, fg_color="transparent")
camera_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")  # sticky nsew biar expand penuh

frame_camera = ctk.CTkLabel(camera_frame, text="")
frame_camera.pack(padx=10, pady=10, expand=True, fill="both")  # expand dan fill both!

# Log Frame
log_frame = ctk.CTkFrame(content_frame, corner_radius=20, fg_color="#1A1A1A")  # abu gelap
log_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")  # sticky nsew biar sejajar kamera

log_title = ctk.CTkLabel(log_frame, text="Log History", font=("Arial", 24, "bold"), text_color="white")
log_title.pack(pady=(20, 10))

log_textbox = ctk.CTkTextbox(log_frame, width=400, height=500, font=("Arial", 16),
                             fg_color="#1A1A1A",  # warna chatbox
                             text_color="white",
                             scrollbar_button_color="grey20",
                             scrollbar_button_hover_color="grey40")
log_textbox.pack(padx=20, pady=(0, 20), expand=True, fill="both")  # padding bawah 20px, biar tombol gak nempel!



# Tombol Snapshot
def take_snapshot():
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with lock:
        snapshot_frame = frame.copy()
    if not os.path.exists("snapshots"):
        os.makedirs("snapshots")
    filepath = os.path.join("snapshots", f"snapshot_{now}.png")
    cv2.imwrite(filepath, snapshot_frame)
    log_textbox.insert("end", f"[{now}] Snapshot disimpan!\n")
    log_textbox.see("end")

snapshot_button = ctk.CTkButton(log_frame, text="Take Snapshot", command=take_snapshot,
                                font=("Arial", 18),
                                fg_color="#333333",  # warna tombol gelap
                                hover_color="#555555",  # saat hover jadi lebih terang
                                text_color="white")
snapshot_button.pack(pady=(0, 20))  # jarak bawah supaya ga nempel

# Update Frame Kamera
def update_frame():
    global frame

    if frame is not None:
        with lock:
            temp_frame = frame.copy()

        # YOLO Helmet Detection
        yolo_results = helmet_model(temp_frame)
        #yolo_results = helmet_model(temp_frame, device=0)  # 0 = GPU
        rgb_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(rgb_frame)

        helmet_boxes = []
        helmet_labels = []
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = result.names[int(box.cls[0])]

                if conf > 0.5:
                    width = x2 - x1
                    height = y2 - y1
                    if width > temp_frame.shape[1] * 0.9 or height > temp_frame.shape[0] * 0.9:
                        continue

                    helmet_boxes.append((x1, y1, x2, y2))
                    helmet_labels.append(label)
                    cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(temp_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Face Detection and Recognition
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = temp_frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                fw = int(bboxC.width * w)
                fh = int(bboxC.height * h)
                x, y = max(0, x), max(0, y)
                fw = min(fw, temp_frame.shape[1] - x)
                fh = min(fh, temp_frame.shape[0] - y)

                face_roi = temp_frame[y:y+fh, x:x+fw]
                rgb_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                encodings = face_recognition.face_encodings(rgb_face_roi)
                name = "Unknown"
                if encodings:
                    face_descriptor = encodings[0]
                    distances = face_recognition.face_distance(known_faces, face_descriptor)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        if min_dist < 0.6:
                            idx = np.argmin(distances)
                            name = known_names[idx]

                face_box = (x, y, x + fw, y + fh)
                helmet_detected = False
                wrong_helmet = False

                for i, (hx1, hy1, hx2, hy2) in enumerate(helmet_boxes):
                    iou_val = iou(face_box, (hx1, hy1, hx2, hy2))
                    if iou_val > 0.1 or is_helmet_above_face(face_box, (hx1, hy1, hx2, hy2)):
                        helmet_detected = True
                        label = helmet_labels[i]
                        helm_owner = label.replace("Helm ", "").strip()
                        if name != "Unknown" and helm_owner != name:
                            wrong_helmet = True
                        break

                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                status_text = "Helm On" if helmet_detected else "Helm Off"
                display_text = f"{name} - {status_text}"
                if wrong_helmet:
                    display_text = f"{name} - Helm Tidak Sesuai"

                log_textbox.insert("end", f"[{timestamp}] {display_text}\n")
                log_textbox.see("end")

                color = (0, 255, 0) if helmet_detected else (0, 0, 255)
                text_color = (0, 0, 255) if wrong_helmet else color
                cv2.rectangle(temp_frame, (x, y), (x + fw, y + fh), color, 2)
                cv2.putText(temp_frame, display_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        img = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        img = img.resize((900, 650))
        imgtk = ImageTk.PhotoImage(image=img)
        frame_camera.imgtk = imgtk
        frame_camera.configure(image=imgtk)

    app.after(10, update_frame)

# Start update
update_frame()

# Start app
app.mainloop()
