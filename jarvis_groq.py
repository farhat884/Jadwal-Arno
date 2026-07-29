import requests
from datetime import datetime

class GroqJarvis:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        
        # 1. Tempat menyimpan memori/riwayat percakapan
        self.chat_history = []  

    def think(self, user_text, current_emotion="Netral"):
        # Ambil tanggal dan tahun real-time dari sistem
        sekarang = datetime.now()
        tanggal_hari_ini = sekarang.strftime("%d %B %Y")

        system_instruction = (
            f"Hari ini adalah tanggal {tanggal_hari_ini}. "
            "Kamu adalah JARVIS, asisten AI pribadi yang sangat cerdas, loyal, dan sopan. "
            "Selalu panggil pengguna dengan sebutan 'Tuan'. "
            "Ingat baik-baik seluruh nama dan informasi yang pernah disampaikan Tuan dalam riwayat percakapan. "
            "Jawablah secara singkat, jelas, dan alami (1-2 kalimat saja) dalam Bahasa Indonesia."
        )

        # 2. Susun daftar pesan lengkap dengan riwayat sebelumnya
        messages = [{"role": "system", "content": system_instruction}]
        
        # Masukkan seluruh percakapan terdahulu
        for msg in self.chat_history:
            messages.append(msg)

        # Masukkan perintah baru dari Tuan
        user_content = f"[Kondisi Ekspresi Tuan: {current_emotion}]\nPerintah Tuan: {user_text}"
        messages.append({"role": "user", "content": user_content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages, # <--- Mengirim seluruh riwayat
            "max_tokens": 150,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                reply = data['choices'][0]['message']['content'].strip()
                
                # 3. Simpan percakapan baru ke memori
                self.chat_history.append({"role": "user", "content": user_text})
                self.chat_history.append({"role": "assistant", "content": reply})

                # Batasi memori (misal simpan 10 percakapan terakhir saja agar tidak terlalu berat)
                if len(self.chat_history) > 10:
                    self.chat_history = self.chat_history[-10:]

                return reply
            else:
                return "Maaf Tuan, terjadi kendala pada respon server Groq."
        except Exception as e:
            return "Maaf Tuan, koneksi ke server terputus."