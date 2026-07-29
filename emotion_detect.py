import os
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort
import tkinter as tk
from PIL import Image, ImageTk
import csv
import time
from datetime import datetime

# ---------------------------------------------------------
# 1. SETUP FILE CSV UNTUK LOGGING
# ---------------------------------------------------------
LOG_FILE = "log_ekspresi.csv"

# Buat file CSV beserta header-nya jika file belum ada
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Ekspresi", "Kepercayaan (%)"])
    print(f"File log baru dibuat: {LOG_FILE}")

# Variabel Kontrol Interval Log (Mencatat setiap 1.0 detik)
LOG_INTERVAL = 1.0  
last_log_time = 0

# ---------------------------------------------------------
# 2. UNDUH MODEL ONNX
# ---------------------------------------------------------
FACE_MODEL_URL = "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx"
FACE_MODEL_PATH = "version-RFB-320.onnx"

EMO_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
EMO_MODEL_PATH = "emotion-ferplus-8.onnx"

if not os.path.exists(FACE_MODEL_PATH):
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)

if not os.path.exists(EMO_MODEL_PATH):
    urllib.request.urlretrieve(EMO_MODEL_URL, EMO_MODEL_PATH)

EMOTIONS = ["Netral", "Senang", "Kaget", "Sedih", "Marah", "Jijik", "Takut", "Sinir/Hina"]

# ---------------------------------------------------------
# 3. INISIALISASI SESSION ONNX
# ---------------------------------------------------------
face_session = ort.InferenceSession(FACE_MODEL_PATH)
face_input_name = face_session.get_inputs()[0].name

emo_session = ort.InferenceSession(EMO_MODEL_PATH)
emo_input_name = emo_session.get_inputs()[0].name

def generate_priors():
    feature_maps = [[30, 40], [15, 20], [8, 10], [4, 5]]
    shrinkage = [8, 16, 32, 64]
    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    priors = []
    for k, (fm_h, fm_w) in enumerate(feature_maps):
        scale_w = 320 / shrinkage[k]
        scale_h = 240 / shrinkage[k]
        for i in range(fm_h):
            for j in range(fm_w):
                cx = (j + 0.5) / scale_w
                cy = (i + 0.5) / scale_h
                for min_box in min_boxes[k]:
                    w = min_box / 320.0
                    h = min_box / 240.0
                    priors.append([cx, cy, w, h])
    return np.array(priors, dtype=np.float32)

PRIORS = generate_priors()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# ---------------------------------------------------------
# 4. BUKA KAMERA & JENDELA TKINTER
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat dibuka!")
    exit()

root = tk.Tk()
root.title("Deteksi Wajah & Ekspresi + CSV Logging")

video_label = tk.Label(root)
video_label.pack()

def update_frame():
    global last_log_time
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    orig_h, orig_w, _ = frame.shape

    # Preprocessing Wajah
    img_resized = cv2.resize(frame, (320, 240))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32)
    img_np = (img_np - 127.0) / 128.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    # Inferensi Deteksi Wajah
    confidences, boxes = face_session.run(None, {face_input_name: img_np})
    confidences = confidences[0]
    boxes = boxes[0]

    threshold = 0.7
    scores = confidences[:, 1]
    mask = scores > threshold

    scores_filtered = scores[mask]
    boxes_filtered = boxes[mask]
    priors_filtered = PRIORS[mask]

    if len(scores_filtered) > 0:
        center_x = boxes_filtered[:, 0] * 0.1 * priors_filtered[:, 2] + priors_filtered[:, 0]
        center_y = boxes_filtered[:, 1] * 0.1 * priors_filtered[:, 3] + priors_filtered[:, 1]
        w = np.exp(boxes_filtered[:, 2] * 0.2) * priors_filtered[:, 2]
        h = np.exp(boxes_filtered[:, 3] * 0.3) * priors_filtered[:, 3]

        x1 = np.clip((center_x - w / 2.0) * orig_w, 0, orig_w)
        y1 = np.clip((center_y - h / 2.0) * orig_h, 0, orig_h)
        x2 = np.clip((center_x + w / 2.0) * orig_w, 0, orig_w)
        y2 = np.clip((center_y + h / 2.0) * orig_h, 0, orig_h)

        for i in range(len(scores_filtered)):
            ix1, iy1, ix2, iy2 = int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])

            face_crop = frame[iy1:iy2, ix1:ix2]
            emo_label_text = "Menganalisis..."

            if face_crop.size > 0:
                gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                resized_crop = cv2.resize(gray_crop, (64, 64))
                img_emo = resized_crop.astype(np.float32)
                img_emo = np.expand_dims(img_emo, axis=(0, 1))

                emo_res = emo_session.run(None, {emo_input_name: img_emo})[0]
                emo_probs = softmax(emo_res[0])
                emo_idx = np.argmax(emo_probs)
                emo_score = emo_probs[emo_idx]

                nama_ekspresi = EMOTIONS[emo_idx]
                persen_akurasi = emo_score * 100
                emo_label_text = f"{nama_ekspresi} ({persen_akurasi:.0f}%)"

                # --- LOGGING KE FILE CSV (Setiap 1 Detik) ---
                current_time = time.time()
                if current_time - last_log_time >= LOG_INTERVAL:
                    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Append baris baru ke file CSV
                    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([waktu_sekarang, nama_ekspresi, f"{persen_akurasi:.1f}"])
                    
                    last_log_time = current_time

            # Visualisasi GUI
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
            cv2.rectangle(frame, (ix1, max(0, iy1 - 30)), (ix1 + 220, max(30, iy1)), (0, 0, 0), -1)
            cv2.putText(frame, f"Ekspresi: {emo_label_text}", (ix1 + 5, max(20, iy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Render Frame ke Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    video_label.img_tk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, update_frame)

def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
root.mainloop()