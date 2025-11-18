import streamlit as st
import io
import textwrap
from docx import Document
from google import genai
from google.genai import types
from pydub import AudioSegment

# --- Page Configuration ---
st.set_page_config(page_title="Gemini TTS Ultimate", page_icon="🎧")

# --- Helper Functions ---
def get_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    else:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()

def generate_audio(client, text_chunks, model_id, voice_name):
    """Generates audio, stitches it properly, and exports as MP3."""
    
    # Standard Gemini TTS 2.5 audio settings (usually 24kHz mono PCM)
    SAMPLE_RATE = 24000 
    
    combined_audio = AudioSegment.empty()
    progress_bar = st.progress(0, text="Starting conversion...")
    total_chunks = len(text_chunks)
    
    for i, chunk in enumerate(text_chunks):
        try:
            # API Call
            response = client.models.generate_content(
                model=model_id,
                contents=chunk,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name
                            )
                        )
                    )
                )
            )
            
            # Collect the raw binary data for this chunk
            chunk_data = b""
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        chunk_data += part.inline_data.data
            
            if chunk_data:
                # Import raw PCM data (Gemini output is raw PCM, 1 channel, 16-bit, 24000Hz)
                # If the API changes to return WAV headers, pydub usually detects it if using from_file
                # But for raw streams, we use from_raw
                segment = AudioSegment.from_file(
                    io.BytesIO(chunk_data), 
                    format="wav" # Gemini API usually wraps it in a WAV container now
                )
                combined_audio += segment
            
            # Update Progress
            percent = int(((i + 1) / total_chunks) * 100)
            progress_bar.progress(percent / 100, text=f"Converting part {i+1}/{total_chunks}...")
            
        except Exception as e:
            st.error(f"Error on chunk {i+1}: {str(e)}")
            return None

    progress_bar.progress(1.0, text="Finalizing MP3...")
    
    # Export to MP3 buffer
    mp3_io = io.BytesIO()
    combined_audio.export(mp3_io, format="mp3", bitrate="192k")
    mp3_io.seek(0)
    return mp3_io

# --- Main Layout ---
st.title("🎧 Gemini 2.5 TTS (MP3 Edition)")
st.markdown("Optimized for fewer API calls + Proper MP3 Output")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google API Key", type="password")
    voice = st.selectbox("Voice", ["Kore", "Puck", "Charon", "Fenrir", "Aoede"])

uploaded_file = st.file_uploader("Upload text or docx", type=['txt', 'docx'])

if uploaded_file and api_key:
    text_content = get_text_from_file(uploaded_file)
    st.caption(f"Character count: {len(text_content)}")
    
    # --- OPTIMIZATION STRATEGY ---
    # You have 15 calls. We maximize usage by sending HUGE chunks.
    # We use 4500 characters (Gemini max safe is ~5000).
    # This allows ~67,000 characters (approx 30 pages) per day with 15 calls.
    chunks = textwrap.wrap(text_content, width=4500, break_long_words=False)
    
    estimated_calls = len(chunks)
    st.info(f"This file will use {estimated_calls} of your 15 daily API calls.")

    if st.button("Generate MP3"):
        client = genai.Client(api_key=api_key)
        
        if estimated_calls > 15:
            st.warning("⚠️ This file is too big for your 15 daily calls! It might fail halfway.")
        
        with st.spinner("Generating..."):
            mp3_data = generate_audio(client, chunks, "gemini-2.5-flash-preview-tts", voice)
            
            if mp3_data:
                st.success("Done!")
                st.audio(mp3_data, format="audio/mp3")
                st.download_button(
                    label="Download MP3",
                    data=mp3_data,
                    file_name=f"{uploaded_file.name.split('.')[0]}.mp3",
                    mime="audio/mp3"
                )
