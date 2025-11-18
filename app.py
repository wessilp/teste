import streamlit as st
import io
import textwrap
from docx import Document
from google import genai
from google.genai import types

# --- Page Configuration ---
st.set_page_config(page_title="Gemini 2.5 TTS", page_icon="🗣️")

# --- Helper Functions ---
def get_text_from_file(uploaded_file):
    """Extracts text from uploaded .txt or .docx file."""
    if uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    else:
        # Assume txt
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()

def generate_audio_chunks(client, chunks, model_id, voice_name):
    """Generates audio for each text chunk and updates the progress bar."""
    audio_parts = []
    progress_bar = st.progress(0, text="Starting conversion...")
    
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        try:
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
            
            # Extract binary audio data
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        audio_parts.append(part.inline_data.data)
            
            # Update Progress
            percent = int(((i + 1) / total_chunks) * 100)
            progress_bar.progress(percent / 100, text=f"Converting chunk {i+1} of {total_chunks}...")
            
        except Exception as e:
            st.error(f"Error on chunk {i+1}: {str(e)}")
            return None

    progress_bar.progress(1.0, text="Conversion Complete!")
    return b"".join(audio_parts)

# --- Main App Layout ---
st.title("🗣️ Gemini 2.5 Flash TTS")
st.markdown("Upload a document, and Gemini will read it to you using the new 2.5 Flash Preview model.")

# Sidebar for Settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google API Key", type="password", help="Get this from Google AI Studio")
    voice = st.selectbox("Select Voice", ["Kore", "Puck", "Charon", "Fenrir", "Aoede"], index=0)
    model_id = "gemini-2.5-flash-preview-tts"
    st.caption(f"Using Model: `{model_id}`")

# File Uploader
uploaded_file = st.file_uploader("Choose a file (.txt or .docx)", type=['txt', 'docx'])

if uploaded_file and api_key:
    # Preview Text
    with st.expander("Preview Text Content"):
        try:
            text_content = get_text_from_file(uploaded_file)
            st.text(text_content[:1000] + ("..." if len(text_content) > 1000 else ""))
        except Exception as e:
            st.error(f"Error reading file: {e}")
            text_content = ""

    # Convert Button
    if st.button("Generate Audio"):
        if not text_content.strip():
            st.warning("File is empty!")
        else:
            client = genai.Client(api_key=api_key)
            
            # Split text to manage progress and API limits
            # Chunks of ~1000 chars are safe and give good progress bar feedback
            chunks = textwrap.wrap(text_content, width=1000, break_long_words=False)
            
            with st.spinner("Initializing Gemini..."):
                audio_data = generate_audio_chunks(client, chunks, model_id, voice)
            
            if audio_data:
                st.success("Audio Generated Successfully!")
                
                # Audio Player
                st.audio(audio_data, format="audio/wav")
                
                # Download Button
                st.download_button(
                    label="Download Audio (.wav)",
                    data=audio_data,
                    file_name=f"{uploaded_file.name.split('.')[0]}.wav",
                    mime="audio/wav"
                )

elif uploaded_file and not api_key:
    st.info("Please enter your Google API Key in the sidebar to proceed.")
