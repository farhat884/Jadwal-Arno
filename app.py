import streamlit as st
import json
import os
import pandas as pd

st.set_page_config(page_title="JARVIS Command Center", page_icon="🤖", layout="wide")

# Styling Sci-Fi Sederhana
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #00ffcc; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 J.A.R.V.I.S Dashboard Agenda")
st.caption("Pusat Kontrol & Catatan Jadwal Suara Real-time")
st.markdown("---")

FILE_JADWAL = "jadwal.json"

def load_data():
    if not os.path.exists(FILE_JADWAL):
        return []
    try:
        with open(FILE_JADWAL, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

data = load_data()

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📋 Daftar Jadwal & Catatan Suara")
    if len(data) == 0:
        st.info("Belum ada jadwal. Ucapkan kalimat seperti: *'Jarvis catat jadwal rapat besok jam 10'*")
    else:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

with col2:
    st.subheader("⚙️ Kontrol Data")
    st.write(f"Total Jadwal: **{len(data)}**")
    
    if st.button("🔄 Refresh Tampilan", use_container_width=True):
        st.rerun()
        
    if st.button("🗑️ Hapus Semua Jadwal", use_container_width=True):
        if os.path.exists(FILE_JADWAL):
            os.remove(FILE_JADWAL)
            st.success("Semua jadwal berhasil dihapus!")
            st.rerun()