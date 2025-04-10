import streamlit as st
import whisper
import tempfile

st.title("🎧 Deutsch Audio-Transkription")

# Language selector
language = st.selectbox("🌐 Select transcription language:", ["de", "tr"], format_func=lambda x: "German 🇩🇪" if x == "de" else "Turkish 🇹🇷")

# File uploader - only audio types
audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file)

    with st.spinner(f"Transcribing in {language.upper()}... ⏳"):
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.name) as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name

        # Load model on CPU
        model = whisper.load_model("medium", device="cpu")

        # Transcribe with selected language
        result = model.transcribe(tmp_path, language=language)

        st.subheader("📝 Transcription")
        st.write(result["text"])