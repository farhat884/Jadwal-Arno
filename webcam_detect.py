import os
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort
import tkinter as tk
from PIL import Image, ImageTk

# 1. Pastikan Model ONNX Terunduh
MODEL_URL = "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx"
MODEL_PATH = "version-RFB-320.onnx"

if not os.path.exists(MODEL_PATH):
    print("Mengunduh model ONNX pendeteksi wajah (~1.2 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# 2. Generator Priors Bounding Box
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

# Load Session ONNX
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# 3. Buka Kamera Laptop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera laptop tidak dapat dibuka!")
    exit()

# 4. Buat Jendela GUI dengan Tkinter (Bypass cv2.imshow)
root = tk.Tk()
root.title("Deteksi Wajah Real-Time (Python 3.14 GUI)")
root.protocol("WM_DELETE_WINDOW", lambda: on_closing())

# Label untuk menampung video stream
video_label = tk.Label(root)
video_label.pack()

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    orig_h, orig_w, _ = frame.shape

    # Preprocessing frame untuk ONNX
    img_resized = cv2.resize(frame, (320, 240))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32)
    img_np = (img_np - 127.0) / 128.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    # Inferensi ONNX
    confidences, boxes = session.run(None, {input_name: img_np})
    confidences = confidences[0]
    boxes = boxes[0]

    # Filter Akurasi (70%)
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

        # Gambar Kotak Merah & Label
        for i in range(len(scores_filtered)):
            pt1 = (int(x1[i]), int(y1[i]))
            pt2 = (int(x2[i]), int(y2[i]))
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            
            label = f"Wajah: {scores_filtered[i]*100:.0f}%"
            cv2.putText(frame, label, (int(x1[i]), max(20, int(y1[i]) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Konversi BGR OpenCV ke format Gambar Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    video_label.img_tk = img_tk
    video_label.configure(image=img_tk)

    # Ulangi setiap 10ms (Real-Time)
    root.after(10, update_frame)

def on_closing():
    cap.release()
    root.destroy()

print("Kamera berhasil terhubung!")
print("Tutup jendela aplikasi untuk menghentikan program.")

# Jalankan loop GUI
update_frame()
root.mainloop()