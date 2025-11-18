import streamlit as st
import time
import io
import wave
import textwrap
from docx import Document
from google import genai
from google.genai import types

# --- ULTRA QUANTITY CONFIGURATION ---
# Based on your data: Output Limit = 16,384 tokens.
# 16k Audio tokens is approx 19-20 minutes of speech.
# 20 mins of speech is roughly 18,000 to 20,000 characters.
# We set this to 18,500 to maximize the call without hitting the hard cut-off.
CHUNK_SIZE = 18500 

st.set_page_config(page_title="Gemini 16k-Token TTS", page_icon="🚀")

def get_text(uploaded_file):
    if uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return uploaded_file.getvalue().decode("utf-8")

def stitch_wavs(audio_parts):
    if not audio_parts: return None
    output = io.BytesIO()
    with wave.open(io.BytesIO(audio_parts[0]), 'rb') as first_wav:
        params = first_wav.getparams()
    with wave.open(output, 'wb') as out_wav:
        out_wav.setparams(params)
        for part in audio_parts:
            with wave.open(io.BytesIO(part), 'rb') as w:
                out_wav.writeframes(w.readframes(w.getnframes()))
    output.seek(0)
    return output

st.title("🚀 Gemini 2.5: 16k Token Limit")
st.markdown("Targeting **~20 minutes of audio per call**.")

# Auto-load key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.success("API Key Loaded")
else:
    api_key = st.text_input("Enter Google API Key", type="password")

uploaded_file = st.file_uploader("Upload File", type=['txt', 'docx'])

if uploaded_file and api_key:
    text = get_text(uploaded_file)
    
    # Huge Chunks
    chunks = textwrap.wrap(text, width=CHUNK_SIZE, break_long_words=False)
    
    # Calculations
    total_calls = len(chunks)
    est_duration = total_calls * 19 # approx 19 mins per chunk
    
    st.info(f"""
    **Specs Analysis:**
    - **Output Limit:** 16,384 Tokens (~20 mins audio)
    - **Chunk Size:** {CHUNK_SIZE} characters
    - **Calls Needed:** {total_calls} / 15 available
    - **Total Audio:** ~{est_duration} minutes ({round(est_duration/60, 1)} hours)
    """)
    
    if total_calls > 15:
        st.warning(f"⚠️ You need {total_calls} calls. The first 15 will generate ~5 hours of audio.")

    if st.button("Generate (Max Capacity)"):
        client = genai.Client(api_key=api_key)
        audio_parts = []
        bar = st.progress(0, text="Starting...")
        
        try:
            for i, chunk in enumerate(chunks):
                if i >= 15:
                    st.error("🛑 Daily limit of 15 calls reached.")
                    break

                # API Call
                response = client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=chunk,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Kore"
                                )
                            )
                        )
                    )
                )
                
                # Extract Data
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            audio_parts.append(part.inline_data.data)
                
                bar.progress((i + 1) / len(chunks), text=f"Processing Part {i+1}/{len(chunks)} (~19 min chunk)")
                
            final_wav = stitch_wavs(audio_parts)
            st.success("Conversion Complete!")
            st.audio(final_wav, format='audio/wav')
            st.download_button("Download WAV", final_wav, file_name=f"{uploaded_file.name}.wav", mime="audio/wav")

        except Exception as e:
            st.error(f"Error: {e}")
