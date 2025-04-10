import streamlit as st
import whisper
import tempfile
import datetime
import traceback

st.title("🎧 Multilingual Audio Transcription with Whisper")

# Language selector
target_language = st.selectbox(
    "🌐 Select transcription language:",
    ["de", "tr"],
    format_func=lambda x: "German 🇩🇪" if x == "de" else "Turkish 🇹🇷"
)

# File uploader
audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file)
    
    with st.spinner("Transcribing... ⏳"):
        try:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name

            # Load smaller Whisper model (change to "tiny" or "base" for Streamlit Cloud)
            model = whisper.load_model("base", device="cpu")

            # Transcribe the audio
            result = model.transcribe(tmp_path, language=target_language)

            # Show transcription
            st.subheader("📝 Transcription")
            st.write(result["text"])

            # Log usage
            with open("usage_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"{datetime.datetime.now()}, {audio_file.name}, {target_language}\n")

        except Exception as e:
            st.error("❌ Transcription failed!")
            st.code(traceback.format_exc())
