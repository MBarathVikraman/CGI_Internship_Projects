import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import tempfile
import os
import queue
from io import BytesIO

from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text
from pydub import AudioSegment
from langdetect import detect
import azure.cognitiveservices.speech as speechsdk
import sounddevice as sd
import soundfile as sf
import numpy as np
import requests

# ====================
# Azure credentials
# ====================
AZURE_TRANSLATOR_KEY = ""
AZURE_TRANSLATOR_ENDPOINT = ""
AZURE_TRANSLATOR_REGION = ""
AZURE_SPEECH_KEY = ""
AZURE_SPEECH_REGION = ""

# ====================
# Translator
# ====================
def translate_text_azure(text, target_lang="en", source_lang=None):
    if source_lang and source_lang != "auto":
        url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate?api-version=3.0&from={source_lang}&to={target_lang}"
    else:
        url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate?api-version=3.0&to={target_lang}"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-type": "application/json"
    }
    body = [{"text": text}]
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()[0]["translations"][0]["text"]

# ====================
# File reading
# ====================
def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext in [".doc", ".docx"]:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".pdf":
        return extract_pdf_text(path)
    else:
        raise ValueError("Unsupported file format.")

# ====================
# Audio Transcription
# ====================
def azure_transcribe(audio_path, language="en-US"):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = language
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = threading.Event()
    results = []

    def recognized(evt):
        if evt.result.text:
            results.append(evt.result.text)

    def stop(evt):
        done.set()

    recognizer.recognized.connect(recognized)
    recognizer.session_stopped.connect(stop)
    recognizer.canceled.connect(stop)

    recognizer.start_continuous_recognition()
    done.wait()
    recognizer.stop_continuous_recognition()

    return " ".join(results)

# ====================
# GUI
# ====================
class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Azure Translator & Real-Time Speech")

        # Language options
        self.languages = {
            "Auto Detect": "auto",
            "English": "en",
            "French": "fr",
            "German": "de",
            "Spanish": "es",
            "Italian": "it",
            "Chinese (Simplified)": "zh-Hans",
            "Japanese": "ja",
            "Hindi": "hi"
        }

        # Text display
        self.text_box = scrolledtext.ScrolledText(root, width=100, height=20)
        self.text_box.pack(padx=10, pady=10)

        # Language selectors
        lang_frame = tk.Frame(root)
        lang_frame.pack(pady=5)

        tk.Label(lang_frame, text="Source Language:").grid(row=0, column=0)
        self.source_lang_var = tk.StringVar(value="Auto Detect")
        tk.OptionMenu(lang_frame, self.source_lang_var, *self.languages.keys()).grid(row=0, column=1)

        tk.Label(lang_frame, text="Target Language:").grid(row=0, column=2)
        self.target_lang_var = tk.StringVar(value="English")
        tk.OptionMenu(lang_frame, self.target_lang_var, *self.languages.keys()).grid(row=0, column=3)

        # Buttons
        frame = tk.Frame(root)
        frame.pack()

        tk.Button(frame, text="Load Document", command=self.load_document).grid(row=0, column=0, padx=5)
        tk.Button(frame, text="Load Audio File", command=self.load_audio).grid(row=0, column=1, padx=5)
        self.start_btn = tk.Button(frame, text="Start Mic", command=self.start_mic)
        self.start_btn.grid(row=0, column=2, padx=5)
        self.stop_btn = tk.Button(frame, text="Stop Mic", command=self.stop_mic, state="disabled")
        self.stop_btn.grid(row=0, column=3, padx=5)

        self.is_recording = False
        self.audio_queue = queue.Queue()

    def get_lang_codes(self):
        source_code = self.languages[self.source_lang_var.get()]
        target_code = self.languages[self.target_lang_var.get()]
        return source_code, target_code

    def load_document(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Documents", "*.txt *.pdf *.doc *.docx")
        ])
        if not path:
            return
        try:
            text = read_file(path)
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, "Original:\n" + text + "\n\nTranslating...")
            self.root.update()
            source, target = self.get_lang_codes()
            translation = translate_text_azure(text, target, None if source == "auto" else source)
            self.text_box.insert(tk.END, "\nTranslation:\n" + translation)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_audio(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Audio", "*.wav *.mp3 *.m4a")
        ])
        if not path:
            return
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                audio = AudioSegment.from_file(path)
                audio.export(temp.name, format="wav")
            source, target = self.get_lang_codes()
            transcription_lang = "en-US" if source == "auto" else source + "-" + source.upper()
            transcript = azure_transcribe(temp.name, language=transcription_lang)
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, "Transcription:\n" + transcript + "\n\nTranslating...")
            self.root.update()
            translation = translate_text_azure(transcript, target, None if source == "auto" else source)
            self.text_box.insert(tk.END, "\nTranslation:\n" + translation)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_mic(self):
        self.is_recording = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        threading.Thread(target=self.mic_recording, daemon=True).start()

    def stop_mic(self):
        self.is_recording = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def mic_recording(self):
        source, target = self.get_lang_codes()
        transcription_lang = "en-US" if source == "auto" else source + "-" + source.upper()

        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_recognition_language = transcription_lang
        push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)

        done = threading.Event()

        def recognized(evt):
            if evt.result.text:
                text = evt.result.text
            # Schedule safe update in main thread
                self.root.after(0, lambda: self.update_transcript(text, source, target))

        def stop(evt):
            done.set()

        recognizer.recognized.connect(recognized)
        recognizer.session_stopped.connect(stop)
        recognizer.canceled.connect(stop)

        recognizer.start_continuous_recognition()

        def callback(indata, frames, time, status):
            if not self.is_recording:
                raise sd.CallbackStop()
            audio_bytes = indata.copy().tobytes()
            push_stream.write(audio_bytes)

        with sd.InputStream(samplerate=16000, channels=1, dtype='int16', callback=callback):
            while self.is_recording:
                sd.sleep(100)

        push_stream.close()
        # Wait for recognition to finish processing
        done.wait()
        recognizer.stop_continuous_recognition()
    def update_transcript(self, text, source, target):
        self.text_box.insert(tk.END, f"\nOriginal: {text}")
        self.text_box.see(tk.END)
        try:
            translation = translate_text_azure(
                text,
                target,
                None if source == "auto" else source
            )
            self.text_box.insert(tk.END, f"\nTranslated: {translation}\n")
            self.text_box.see(tk.END)
        except Exception as e:
            self.text_box.insert(tk.END, f"\nTranslation error: {e}")
            self.text_box.see(tk.END)


# ====================
# Run
# ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
