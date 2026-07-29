import os
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

# 1. Auto-download model ONNX ringan (Ultra-Light Face Detector ~1.2 MB)
MODEL_URL = "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx"
MODEL_PATH = "version-RFB-320.onnx"

if not os.path.exists(MODEL_PATH):
    print("Mengunduh model ONNX pendeteksi wajah (~1.2 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model berhasil diunduh!")

# 2. Pembuatan Anchor Boxes untuk post-processing
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

# 3. Fungsi Utama Deteksi Wajah
def detect_faces(image_path, output_path="hasil_deteksi.jpg", threshold=0.7):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' tidak ditemukan!")
        return

    # Load Gambar pakai PIL
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Preprocessing (Resize ke 320x240 & Normalisasi array)
    img_resized = orig_img.resize((320, 240))
    img_np = np.array(img_resized, dtype=np.float32)
    img_np = (img_np - 127.0) / 128.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    # Jalankan Inferensi dengan ONNX Runtime
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    confidences, boxes = session.run(None, {input_name: img_np})

    confidences = confidences[0]
    boxes = boxes[0]

    # Filter hasil berdasarkan threshold keyakinan
    scores = confidences[:, 1]
    mask = scores > threshold

    scores = scores[mask]
    boxes = boxes[mask]
    priors = PRIORS[mask]

    if len(scores) == 0:
        print("Tidak ada wajah yang terdeteksi.")
        return

    # Dekode koordinat kotak wajah (Bounding Boxes)
    center_x = boxes[:, 0] * 0.1 * priors[:, 2] + priors[:, 0]
    center_y = boxes[:, 1] * 0.1 * priors[:, 3] + priors[:, 1]
    w = np.exp(boxes[:, 2] * 0.2) * priors[:, 2]
    h = np.exp(boxes[:, 3] * 0.3) * priors[:, 3]

    x1 = np.clip((center_x - w / 2.0) * orig_w, 0, orig_w)
    y1 = np.clip((center_y - h / 2.0) * orig_h, 0, orig_h)
    x2 = np.clip((center_x + w / 2.0) * orig_w, 0, orig_w)
    y2 = np.clip((center_y + h / 2.0) * orig_h, 0, orig_h)

    # Gambar kotak di atas gambar asli pakai PIL
    draw = ImageDraw.Draw(orig_img)
    for i in range(len(scores)):
        box = [x1[i], y1[i], x2[i], y2[i]]
        draw.rectangle(box, outline="red", width=4)
        print(f"Wajah #{i+1} terdeteksi! Akurasi: {scores[i]*100:.1f}%")

    # Simpan Hasil
    orig_img.save(output_path)
    print(f"\nBerhasil! Gambar hasil deteksi disimpan di: {output_path}")

# --- JALANKAN PROGRAM ---
if __name__ == "__main__":
    # Ganti "input.jpg" dengan nama file gambar yang mau kamu tes
    nama_gambar = "input.jpg" 
    detect_faces(nama_gambar, threshold=0.6)