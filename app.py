import streamlit as st
import json
import os
import time
import subprocess
import tempfile
import asyncio
import pandas as pd
import cv2
import numpy as np
import onnxruntime as ort
import speech_recognition as sr
import edge_tts
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Import Groq AI Jarvis
from jarvis_groq import GroqJarvis

# =========================================================
# 1. KONFIGURASI HALAMAN & TEMA SCI-FI
# =========================================================
st.set_page_config(
    page_title="JARVIS Command Center", 
    page_icon="🤖", 
    layout="wide"
)

# Custom CSS Sci-Fi Theme
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #00ffcc !important; }
    .stButton>button {
        background-color: #1a1c23;
        color: #00ffcc;
        border: 1px solid #00ffcc;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #00ffcc;
        color: #0e1117;
    }
    </style>
""", unsafe_allow_html=True)

FILE_JADWAL = "jadwal.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "masukkan_api_key_disini")

# Inisialisasi Resource AI
@st.cache_resource
def get_jarvis():
    return GroqJarvis(api_key=GROQ_API_KEY)

jarvis_brain = get_jarvis()

# Inisialisasi Session State (Agar percakapan tidak terulang/terhapus)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Sistem J.A.R.V.I.S siap, Tuan. Ada yang bisa saya bantu hari ini?"}
    ]

if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "Netral"

# =========================================================
# 2. FUNGSI UTILITY (LOAD/SAVE DATA & GITHUB PUSH)
# =========================================================
def push_ke_github():
    """Mengirim update jadwal ke repository GitHub"""
    try:
        subprocess.run(["git", "add", FILE_JADWAL], check=True)
        commit_msg = f"Update jadwal otomatis: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        st.toast("✅ Auto-sync GitHub Berhasil!", icon="🚀")
    except Exception as e:
        st.sidebar.error(f"Gagal Push Git: {e}")

def load_data():
    if not os.path.exists(FILE_JADWAL):
        return []
    try:
        with open(FILE_JADWAL, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except:
        return []

def simpan_jadwal_otomatis(command_text):
    data = load_data()
    jadwal_baru = {
        "Waktu Input": time.strftime("%d-%m-%Y %H:%M"),
        "Detail Perintah": command_text.capitalize(),
        "Status": "Aktif"
    }
    data.append(jadwal_baru)
    try:
        with open(FILE_JADWAL, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        push_ke_github()
    except Exception as e:
        st.error(f"Gagal menyimpan jadwal: {e}")

# =========================================================
# 3. TEXT TO SPEECH (TTS) STREAMLIT
# =========================================================
async def generate_speech_bytes(text):
    temp_file = os.path.join(tempfile.gettempdir(), f"speech_{int(time.time())}.mp3")
    communicate = edge_tts.Communicate(text, "id-ID-ArdiNeural")
    await communicate.save(temp_file)
    return temp_file

def play_speech(text):
    try:
        audio_path = asyncio.run(generate_speech_bytes(text))
        st.audio(audio_path, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.warning(f"Audio Error: {e}")

# =========================================================
# 4. DETEKSI EMOSI (ONNX)
# =========================================================
EMO_MODEL_PATH = "emotion-ferplus-8.onnx"
EMOTIONS = ["Netral", "Senang", "Kaget", "Sedih", "Marah", "Jijik", "Takut", "Sinir/Hina"]

@st.cache_resource
def load_emo_model():
    import urllib.request
    url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    if not os.path.exists(EMO_MODEL_PATH):
        urllib.request.urlretrieve(url, EMO_MODEL_PATH)
    return ort.InferenceSession(EMO_MODEL_PATH)

def detect_emotion(image_bytes):
    try:
        emo_session = load_emo_model()
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)).astype(np.float32)
        img_emo = np.expand_dims(resized, axis=(0, 1))
        
        emo_input_name = emo_session.get_inputs()[0].name
        emo_outs = emo_session.run(None, {emo_input_name: img_emo})[0][0]
        
        exp_x = np.exp(emo_outs - np.max(emo_outs))
        probs = exp_x / exp_x.sum()
        return EMOTIONS[np.argmax(probs)]
    except Exception as e:
        return "Netral"

# =========================================================
# 5. TAMPILAN DASHBOARD UTAMA
# =========================================================
st.title("🤖 J.A.R.V.I.S Command Center")
st.caption("Pusat Kontrol & Catatan Jadwal Suara Real-time")
st.markdown("---")

# Layout 2 Tab: Tab 1 (Dashboard Agenda & Chat AI), Tab 2 (Kamera & Visi)
tab1, tab2 = st.tabs(["💬 Control Center & Agenda", "📸 Vision & Emotion Scanner"])

# ----------------- TAB 1: AGENDA & CHAT AI -----------------
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Daftar Jadwal & Catatan Suara")
        data = load_data()
        if len(data) == 0:
            st.info("Belum ada jadwal. Ucapkan kalimat seperti: *'Jarvis catat jadwal rapat besok jam 10'*")
        else:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("⚙️ Kontrol Data")
        st.write(f"Total Jadwal Terdaftar: **{len(data)}**")
        st.write(f"Status Emosi Target: **{st.session_state.current_emotion.upper()}**")
        
        if st.button("🔄 Refresh Tampilan", use_container_width=True):
            st.rerun()
            
        if st.button("🗑️ Hapus Semua Jadwal", use_container_width=True):
            if os.path.exists(FILE_JADWAL):
                os.remove(FILE_JADWAL)
                push_ke_github()
                st.success("Semua jadwal berhasil dihapus!")
                st.rerun()

    st.markdown("---")
    st.subheader("💬 Komunikasi Interaktif JARVIS")

    # Tampilkan Riwayat Percakapan
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Kontrol Input Suara & Teks
    col_mic, col_txt = st.columns([1, 4])
    
    user_voice = None
    with col_mic:
        if st.button("🎙️ Bicara (Mic)", width="stretch"):
            try:
                r = sr.Recognizer()
                # Cek apakah PyAudio/Microphone tersedia di sistem
                with sr.Microphone() as source:
                    st.toast("Mendengarkan...", icon="👂")
                    audio = r.listen(source, timeout=5, phrase_time_limit=10)
                    user_voice = r.recognize_google(audio, language="id-ID").lower()
            except AttributeError:
                st.error("⚠️ Input mikrofon fisik hanya tersedia saat aplikasi dijalankan di Komputer Lokal.")
            except Exception as e:
                st.error("Gagal menangkap suara dari mikrofon.")

    with col_txt:
        user_text = st.chat_input("Ketik perintah atau pertanyaan untuk JARVIS...")

    # Gabungkan input mana yang dipakai (Teks atau Suara)
    final_input = user_voice if user_voice else user_text

    if final_input:
        # Simpan & Tampilkan Pesan User
        st.session_state.messages.append({"role": "user", "content": final_input})
        with st.chat_message("user"):
            st.markdown(final_input)

        # Cek jika perintah mengandung instruksi mencatat jadwal
        if any(kata in final_input for kata in ["jadwal", "catat", "ingatkan", "agenda"]):
            simpan_jadwal_otomatis(final_input)

        # Respon dari JARVIS
        with st.chat_message("assistant"):
            with st.spinner("JARVIS sedang memproses..."):
                reply = jarvis_brain.think(final_input, st.session_state.current_emotion)
                st.markdown(reply)
                play_speech(reply)

        # Simpan Respon AI ke Memory Session
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ----------------- TAB 2: KAMERA & VISI -----------------
with tab2:
    st.subheader("📸 Pindai Emosi Pengguna")
    st.caption("Ambil foto wajah untuk memperbarui ekspresi emosi target secara real-time.")
    
    img_input = st.camera_input("Kamera Visi Target")
    
    if img_input:
        detected = detect_emotion(img_input)
        st.session_state.current_emotion = detected
        st.success(f"Emosi berhasil dideteksi: **{detected.upper()}**")
        st.info("Emosi ini akan disesuaikan secara otomatis saat JARVIS merespons percakapanmu.")