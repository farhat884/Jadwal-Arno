import os
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort
import tkinter as tk
from PIL import Image, ImageTk
import speech_recognition as sr
import threading
import time
import queue
import asyncio
import tempfile
import ctypes
import edge_tts
import json
import subprocess  # === [TAMBAHAN]: Untuk menjalankan perintah Git otomatis ===

# =========================================================
# 0. KONFIGURASI AI LOKAL / CLOUD (Groq API)
# =========================================================
from jarvis_groq import GroqJarvis

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "isi_api_key_kamu_lokal_saja")
jarvis_brain = GroqJarvis(api_key=GROQ_API_KEY)

def ask_jarvis_ai(user_text, current_emotion="Netral"):
    return jarvis_brain.think(user_text, current_emotion)

# === [TAMBAHAN STREAMLIT + GITHUB]: FUNGSI AUTO PUSH KE GITHUB ===
FILE_JADWAL = "jadwal.json"

def push_ke_github():
    """Menjalankan perintah Git di latar belakang agar tidak mengganggu GUI/Suara"""
    def task():
        try:
            print("\n[GITHUB AUTO-SYNC]: Memulai proses Push ke GitHub...")
            # 1. Add file jadwal.json
            subprocess.run(["git", "add", FILE_JADWAL], check=True)
            
            # 2. Commit dengan pesan waktu otomatis
            commit_msg = f"Update jadwal otomatis: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            
            # 3. Push ke branch main (sesuaikan jika nama branch kamu 'master')
            subprocess.run(["git", "push", "origin", "main"], check=True)
            
            print("[GITHUB AUTO-SYNC]: Berhasil di-push ke GitHub! Streamlit Cloud akan me-redeploy otomatis.\n")
        except Exception as e:
            print(f"[GITHUB AUTO-SYNC ERROR]: Gagal push ke GitHub -> {e}\n")

    # Jalankan di Thread terpisah agar aplikasi JARVIS tidak freeze/lag
    threading.Thread(target=task, daemon=True).start()

def simpan_jadwal_otomatis(command_text):
    """Mencatat jadwal secara otomatis ke file JSON dan push ke GitHub"""
    try:
        if os.path.exists(FILE_JADWAL):
            with open(FILE_JADWAL, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []

    jadwal_baru = {
        "Waktu Input": time.strftime("%d-%m-%Y %H:%M"),
        "Detail Perintah": command_text.capitalize(),
        "Status": "Aktif"
    }
    
    data.append(jadwal_baru)

    # Simpan ke file lokal
    with open(FILE_JADWAL, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[STREAMLIT SYSTEM]: Data jadwal berhasil dicatat ke '{FILE_JADWAL}'")
    
    # === PANGGUL AUTO PUSH KE GITHUB ===
    push_ke_github()
# =============================================================

# ---------------------------------------------------------
# 1. GLOBAL STATE
# ---------------------------------------------------------
is_listening = True
is_speaking = False
current_expression = "Netral"
has_greeted = False

# ---------------------------------------------------------
# 2. SUARA JARVIS (Microsoft Neural AI - Ardi Indonesia)
# ---------------------------------------------------------
speak_queue = queue.Queue()

async def generate_speech(text, output_file):
    communicate = edge_tts.Communicate(text, "id-ID-ArdiNeural")
    await communicate.save(output_file)

def play_audio_native(file_path):
    mci = ctypes.windll.winmm.mciSendStringW
    mci('close jarvis_voice', None, 0, 0)
    mci(f'open "{file_path}" type mpegvideo alias jarvis_voice', None, 0, 0)
    mci('play jarvis_voice wait', None, 0, 0)
    mci('close jarvis_voice', None, 0, 0)

def speech_worker():
    global is_speaking
    temp_file = os.path.join(tempfile.gettempdir(), "jarvis_speech.mp3")
    
    while True:
        text = speak_queue.get()
        if text is None: break
        try:
            is_speaking = True
            print(f"\n[JARVIS]: {text}")
            
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except: pass

            asyncio.run(generate_speech(text, temp_file))
            play_audio_native(temp_file)
            time.sleep(0.3)
        except Exception as e:
            print(f"[Audio Error]: {e}")
        finally:
            is_speaking = False
            speak_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def jarvis_speak(text):
    speak_queue.put(text)

# ---------------------------------------------------------
# 3. MODUL TELINGA (Dinamis & Anti-Gema)
# ---------------------------------------------------------
def listen_microphone():
    global current_expression, is_listening, is_speaking
    recognizer = sr.Recognizer()
    
    print("\n[SISTEM]: Memulai kalibrasi mikrofon...")
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print("[SISTEM]: Modul pendengar aktif. Silakan bicara.")

        while is_listening:
            if is_speaking:
                time.sleep(0.5)
                continue

            try:
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=15)
                
                if is_speaking: continue
                
                print("[SISTEM]: Memproses suara...")
                command = recognizer.recognize_google(audio_data, language="id-ID").lower()
                print(f"\n[Tuan]: '{command}'")

                if any(k in command for k in ["matikan", "keluar", "shutdown", "berhenti", "close"]):
                    jarvis_speak("Mematikan protokol. Sampai jumpa kembali, Tuan.")
                    time.sleep(3)
                    os._exit(0)

                # Deteksi otomatis kata kunci jadwal
                if any(kata in command for kata in ["jadwal", "catat", "ingatkan", "agenda"]):
                    simpan_jadwal_otomatis(command)

                # Kirim ke LLM
                ai_reply = ask_jarvis_ai(command, current_expression)
                jarvis_speak(ai_reply)

                time.sleep(1)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"[Mic Error]: {e}")
                time.sleep(1)

threading.Thread(target=listen_microphone, daemon=True).start()

# ---------------------------------------------------------
# 4. MODEL ONNX (Wajah & Ekspresi)
# ---------------------------------------------------------
FACE_MODEL_PATH = "version-RFB-320.onnx"
EMO_MODEL_PATH = "emotion-ferplus-8.onnx"

for url, path in [
    ("https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx", FACE_MODEL_PATH),
    ("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx", EMO_MODEL_PATH)
]:
    if not os.path.exists(path): urllib.request.urlretrieve(url, path)

EMOTIONS = ["Netral", "Senang", "Kaget", "Sedih", "Marah", "Jijik", "Takut", "Sinir/Hina"]

face_session = ort.InferenceSession(FACE_MODEL_PATH)
emo_session = ort.InferenceSession(EMO_MODEL_PATH)
face_input_name = face_session.get_inputs()[0].name
emo_input_name = emo_session.get_inputs()[0].name

def generate_priors():
    feature_maps = [[30, 40], [15, 20], [8, 10], [4, 5]]
    shrinkage = [8, 16, 32, 64]
    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    priors = []
    for k, (fm_h, fm_w) in enumerate(feature_maps):
        for i in range(fm_h):
            for j in range(fm_w):
                for min_box in min_boxes[k]:
                    priors.append([(j + 0.5) / (320 / shrinkage[k]), (i + 0.5) / (240 / shrinkage[k]), min_box / 320.0, min_box / 240.0])
    return np.array(priors, dtype=np.float32)

PRIORS = generate_priors()
def softmax(x): e_x = np.exp(x - np.max(x)); return e_x / e_x.sum(axis=-1, keepdims=True)

# ---------------------------------------------------------
# 5. KAMERA & GUI (Tema Futuristik)
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("J.A.R.V.I.S Protocol v2.0")
root.geometry("700x580")
root.configure(bg="#0a0a0a")

header = tk.Label(root, text="J.A.R.V.I.S  V.I.S.I.O.N  S.Y.S.T.E.M", bg="#0a0a0a", fg="#00ffcc", font=("Courier", 16, "bold"))
header.pack(pady=10)

video_label = tk.Label(root, bg="#000000", bd=2, relief="solid")
video_label.pack()

status_label = tk.Label(root, text="STATUS: SCANNING...", bg="#0a0a0a", fg="#00ff00", font=("Courier", 10))
status_label.pack(pady=10)

jarvis_speak("Sistem aktif. Protokol visi dan kecerdasan buatan telah terhubung.")

def update_frame():
    global has_greeted, current_expression
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    orig_h, orig_w, _ = frame.shape
    img_np = np.expand_dims(np.transpose((cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.0) / 128.0, (2, 0, 1)), axis=0)

    confidences, boxes = face_session.run(None, {face_input_name: img_np})
    scores = confidences[0][:, 1]
    mask = scores > 0.7

    if mask.any():
        boxes_filtered, priors_filtered = boxes[0][mask], PRIORS[mask]
        center_x = boxes_filtered[:, 0] * 0.1 * priors_filtered[:, 2] + priors_filtered[:, 0]
        center_y = boxes_filtered[:, 1] * 0.1 * priors_filtered[:, 3] + priors_filtered[:, 1]
        w = np.exp(boxes_filtered[:, 2] * 0.2) * priors_filtered[:, 2]
        h = np.exp(boxes_filtered[:, 3] * 0.3) * priors_filtered[:, 3]

        for i, (cx, cy, bw, bh) in enumerate(zip(center_x, center_y, w, h)):
            x1, y1 = max(0, int((cx - bw / 2) * orig_w)), max(0, int((cy - bh / 2) * orig_h))
            x2, y2 = min(orig_w, int((cx + bw / 2) * orig_w)), min(orig_h, int((cy + bh / 2) * orig_h))
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                img_emo = np.expand_dims(cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32), axis=(0, 1))
                emo_probs = softmax(emo_session.run(None, {emo_input_name: img_emo})[0][0])
                current_expression = EMOTIONS[np.argmax(emo_probs)]

                if not has_greeted:
                    jarvis_speak("Selamat datang kembali, Tuan. Visi telah terkunci pada target.")
                    has_greeted = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 204), 1)
                cv2.rectangle(frame, (x1, y1), (x1+15, y1+15), (0, 255, 204), -1)
                cv2.putText(frame, f"TARGET: {current_expression.upper()}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 204), 2)
                
                status_label.config(text=f"STATUS: TARGET DETECTED | EMOTION: {current_expression.upper()}", fg="#00ffcc")

    else:
        status_label.config(text="STATUS: SEARCHING...", fg="#ff3333")

    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    video_label.img_tk = img_tk
    video_label.configure(image=img_tk)
    root.after(10, update_frame)

def on_closing():
    global is_listening
    is_listening = False
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
root.mainloop()