import tkinter as tk
from tkinter import filedialog
import threading
import queue
import sounddevice as sd
import numpy as np
import whisper
import requests
import librosa
import os

# ----------- Config -----------
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
MODEL_SIZE = "small"
TRANSLATE_URL = "http://localhost:5000/translate"

# ----------- Load Model -----------
model = whisper.load_model(MODEL_SIZE, device="cpu")

# ----------- Queues -----------
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_q.put(indata.copy())

# ----------- Translation -----------
def translate_text(text, detected_lang):
    try:
        selected_lang = source_lang_var.get()
        target_lang = target_lang_var.get()

        if selected_lang == "auto":
            source_lang = detected_lang
        else:
            if selected_lang != detected_lang and len(text.split()) > 3:
                mismatch_ratio = sum(1 for w in text.split() if w.isascii()) / max(1, len(text.split()))
                if mismatch_ratio > 0.7:
                    print("Mismatch too high — using Whisper detected language instead")
                    source_lang = detected_lang
                else:
                    source_lang = selected_lang
            else:
                source_lang = selected_lang

        print(f"Translating from {source_lang} to {target_lang}: {text[:50]}...")
        response = requests.post(TRANSLATE_URL, json={
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }, timeout=10)

        if response.status_code == 200:
            return response.json().get("translatedText", "")
        else:
            return f"[Translation Error] HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return f"[Translation Failed] {e}"

# ----------- Audio Processor Thread -----------
def process_audio_stream():
    audio_buffer = []
    silence_threshold = 30
    silence_count = 0
    min_speech_duration = 1.0
    max_buffer_duration = 2.0

    while app_running:
        try:
            audio_data = audio_q.get(timeout=1)
            audio_buffer.append(audio_data)

            current_rms = np.sqrt(np.mean(audio_data ** 2))
            current_db = 20 * np.log10(current_rms + 1e-10)

            silence_db_threshold = sensitivity_var.get()

            if current_db < silence_db_threshold:
                silence_count += 1
            else:
                silence_count = 0

            current_buffer_duration = len(audio_buffer) * BLOCK_SIZE / SAMPLE_RATE
            should_process = (silence_count >= silence_threshold and audio_buffer) or \
                             (current_buffer_duration >= max_buffer_duration)

            if should_process:
                audio_np = np.concatenate(audio_buffer, axis=0)
                if audio_np.ndim == 2:
                    audio_np = audio_np.flatten()

                audio_duration = len(audio_np) / SAMPLE_RATE

                if audio_duration >= min_speech_duration:
                    try:
                        audio_float = audio_np.astype(np.float32)
                        buffer_rms = np.sqrt(np.mean(audio_float ** 2))
                        buffer_db = 20 * np.log10(buffer_rms + 1e-10)

                        if buffer_db < -40:
                            output_text.set(f"[Skipped] Audio too quiet ({buffer_db:.1f} dB). No speech detected.")
                        else:
                            result = model.transcribe(audio_float)
                            original_text = result["text"].strip()
                            lang = result.get("language", "unknown")

                            if original_text and len(original_text) > 3:
                                translated = translate_text(original_text, lang)
                                output_text.set(f"Detected: {lang}\n\nOriginal: {original_text}\n\nTranslation: {translated}")
                            else:
                                output_text.set(f"[Warning] No meaningful speech detected.\nTranscribed text: '{original_text}'")
                    except Exception as transcribe_error:
                        output_text.set(f"[Transcription Error] {transcribe_error}")

                audio_buffer = []
                silence_count = 0

        except queue.Empty:
            continue
        except Exception as e:
            output_text.set(f"[Error] {e}")
            audio_buffer = []
            silence_count = 0

# ----------- Upload Audio File -----------
def upload_audio_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a")])
    if file_path:
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            duration = len(y) / SAMPLE_RATE

            if duration < 0.1:
                raise ValueError("Audio file too short (< 0.1 seconds)")

            result = model.transcribe(y)
            original_text = result["text"].strip()
            lang = result.get("language", "unknown")

            if not original_text or len(original_text) < 2:
                output_text.set(f"[Warning] No speech detected.\nFile: {os.path.basename(file_path)}\nDuration: {duration:.2f}s")
                return

            translated = translate_text(original_text, lang)
            output_text.set(f"File: {os.path.basename(file_path)}\nDetected: {lang}\n\nOriginal: {original_text}\n\nTranslation: {translated}")

        except Exception as e:
            output_text.set(f"[File Error] {e}\nFile: {os.path.basename(file_path)}")

# ----------- Upload Text File -----------
def upload_text_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text/Doc Files", "*.txt *.pdf *.doc *.docx")])
    if file_path:
        try:
            text = ""
            if file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif file_path.endswith(".pdf"):
                import fitz
                doc = fitz.open(file_path)
                for page in doc:
                    text += page.get_text()
            elif file_path.endswith(".doc") or file_path.endswith(".docx"):
                import docx
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"

            translated = translate_text(text.strip(), "auto")
            output_text.set(f"Original:\n{text.strip()}\n\nTranslation:\n{translated}")
        except Exception as e:
            output_text.set(f"[Text File Error] {e}")

# ----------- Test Translation Service -----------
def test_translation_service():
    try:
        test_text = "Bonjour le monde"
        response = requests.post(TRANSLATE_URL, json={
            "q": test_text,
            "source": "fr",
            "target": "en",
            "format": "text"
        }, timeout=5)
        if response.status_code == 200:
            result = response.json()
            translated = result.get("translatedText", "")
            output_text.set(f"Translation Service Test:\nOriginal: {test_text}\nTranslated: {translated}")
        else:
            output_text.set(f"Translation Service Error: HTTP {response.status_code}")
    except Exception as e:
        output_text.set(f"Translation Service Test Failed: {e}")

# ----------- Listening Control -----------
app_running = False
stream = None

def start_listening():
    global app_running, stream
    app_running = True
    output_text.set("Listening... Speak clearly and then pause to translate.")
    threading.Thread(target=process_audio_stream, daemon=True).start()
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                blocksize=BLOCK_SIZE, callback=audio_callback)
        stream.start()
    except Exception as e:
        output_text.set(f"Mic Error: {e}")

def stop_listening():
    global app_running, stream
    app_running = False
    if stream:
        try:
            stream.stop()
            stream.close()
        except:
            pass
    output_text.set("Stopped.")

# ----------- GUI -----------
root = tk.Tk()
root.title("Real-Time Multilingual Translator")
root.geometry("850x650")

output_text = tk.StringVar()
output_text.set("Click 'Start Listening' to begin")

output_label = tk.Label(root, textvariable=output_text, wraplength=800, justify="left", anchor="nw",
                       font=("Arial", 12), bg="white", relief="sunken", bd=2)
output_label.pack(padx=10, pady=10, fill="both", expand=True)

# Language Selection
lang_frame = tk.Frame(root)
lang_frame.pack(pady=5)

tk.Label(lang_frame, text="Source Language:").pack(side="left", padx=5)
source_lang_var = tk.StringVar(value="auto")
tk.OptionMenu(lang_frame, source_lang_var, "auto", "en", "fr", "de", "es", "hi", "zh", "ja").pack(side="left", padx=5)

tk.Label(lang_frame, text="Target Language:").pack(side="left", padx=5)
target_lang_var = tk.StringVar(value="en")
tk.OptionMenu(lang_frame, target_lang_var, "en", "fr", "de", "es", "hi", "zh", "ja").pack(side="left", padx=5)

# RMS Sensitivity
sensitivity_frame = tk.Frame(root)
sensitivity_frame.pack(pady=5)

tk.Label(sensitivity_frame, text="Mic Sensitivity (Lower = More sensitive):").pack(side="left", padx=5)
sensitivity_var = tk.DoubleVar(value=-35.0)
tk.Scale(sensitivity_frame, from_=-60, to=-10, resolution=1, variable=sensitivity_var,
         orient="horizontal", length=300).pack(side="left", padx=5)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="🎙️ Start Listening", command=start_listening,
          font=("Arial", 10), bg="lightgreen").pack(side="left", padx=5)
tk.Button(btn_frame, text="⏹️ Stop Listening", command=stop_listening,
          font=("Arial", 10), bg="lightcoral").pack(side="left", padx=5)
tk.Button(btn_frame, text="📁 Upload Audio", command=upload_audio_file,
          font=("Arial", 10), bg="lightblue").pack(side="left", padx=5)
tk.Button(btn_frame, text="📄 Upload Text", command=upload_text_file,
          font=("Arial", 10), bg="lightyellow").pack(side="left", padx=5)
tk.Button(btn_frame, text="🔧 Test Translation", command=test_translation_service,
          font=("Arial", 10), bg="lightgray").pack(side="left", padx=5)

instructions = tk.Label(root, text="Instructions: Speak clearly and pause. Adjust mic sensitivity if needed.",
                       font=("Arial", 10), fg="gray")
instructions.pack(pady=5)

# App close handler
def on_closing():
    global app_running, stream
    app_running = False
    if stream:
        try:
            stream.stop()
            stream.close()
        except:
            pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
