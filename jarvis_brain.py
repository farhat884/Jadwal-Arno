import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class NativeJarvisAI:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        print("[JARVIS Brain]: Memuat otak AI lokal ke memori...")
        
        # Otomatis memilih GPU (CUDA) jika ada, atau CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model.to("cpu")
            
        print(f"[JARVIS Brain]: Otak AI berhasil dimuat menggunakan device: {self.device}")

    def think(self, user_text, current_emotion="Netral"):
        system_instruction = (
            "Kamu adalah JARVIS, asisten AI pribadi yang sangat cerdas, loyal, dan sopan. "
            "Selalu panggil pengguna dengan sebutan 'Tuan'. "
            "Jawablah secara singkat dan alami (maksimal 1-2 kalimat) dalam Bahasa Indonesia."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"[Ekspresi Tuan: {current_emotion}]\nPerintah Tuan: {user_text}"}
        ]

        # Format prompt sesuai standar model
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Proses pemikiran AI secara langsung (Generasi Teks)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()