Building a Multilingual Transcription App Using Whisper and Streamlit
🚀 Why I Built This App
In a recent research project, our team was tasked with transcribing hundreds of hours of interviews conducted in German and Turkish. However, due to policy constraints, we were required to use an officially approved tool - Amberscript - for the final transcription. While Amberscript is a great tool, I realized there are moments in research and academic work where having an in-house, open-source transcription solution could provide:
✅ Flexibility to handle different languages
✅ Full control over data (especially for sensitive files)
✅ Zero cost at scale
✅ Customizability to add features like translation, summarization, etc.
That curiosity led me to build this lightweight yet powerful web-based transcription app using OpenAI's Whisper model and Streamlit.
📄 The Code (whisper_app.py)
import streamlit as st
import whisper
import tempfile

st.title("🎷 Multilingual Audio Transcription with Whisper")

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
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.name) as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name

        # Load Whisper model on CPU
        model = whisper.load_model("medium", device="cpu")

        # Transcribe the audio
        result = model.transcribe(tmp_path, language=target_language)

        # Show transcription
        st.subheader("📝 Transcription")
        st.write(result["text"])
🔗 Deploying It on Streamlit Cloud
Create a GitHub repository
Add whisper_app.py and this requirements.txt:

streamlit
openai-whisper
torch
ffmpeg-python
3. Go to streamlit.io/cloud
4. Connect your GitHub repo and deploy
🚀 What's Next?
Add English translation option
Export transcription to .txt or .srt
Add support for other languages
Batch upload

Even if not used for the current project, this app gave me new superpowers. I'm excited to improve and expand it for future research use cases - or even build tools for clinical transcription or education.
GitHub Repo: [your-repo-link-here]
Try the App on Streamlit Cloud: [your-app-link-here]
