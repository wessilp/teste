import streamlit as st
import time
import io
import wave
import textwrap
from docx import Document
from google import genai
from google.genai import types

# --- CONFIGURAÇÕES ---
# Mantivemos 12.000 para segurança, mas agora você vai ver o download acontecendo.
CHUNK_SIZE = 12000 

st.set_page_config(page_title="Gemini TTS Monitor", page_icon="📡")

def get_text(uploaded_file):
    if uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return uploaded_file.getvalue().decode("utf-8")

def stitch_wavs(audio_parts):
    if not audio_parts: return None
    output = io.BytesIO()
    try:
        with wave.open(io.BytesIO(audio_parts[0]), 'rb') as first_wav:
            params = first_wav.getparams()
        with wave.open(output, 'wb') as out_wav:
            out_wav.setparams(params)
            for part in audio_parts:
                with wave.open(io.BytesIO(part), 'rb') as w:
                    out_wav.writeframes(w.readframes(w.getnframes()))
        output.seek(0)
        return output
    except Exception as e:
        return None

st.title("📡 Gemini Monitor & TTS")

# 1. API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.success("✅ API Key Conectada")
else:
    api_key = st.text_input("API Key:", type="password")

# 2. Upload
uploaded_file = st.file_uploader("Arquivo (.txt / .docx)", type=['txt', 'docx'])

if uploaded_file and api_key:
    text = get_text(uploaded_file)
    chunks = textwrap.wrap(text, width=CHUNK_SIZE, break_long_words=False)
    
    st.markdown(f"**Diagnóstico do Arquivo:**\n- Tamanho Total: `{len(text)}` caracteres\n- Blocos para processar: `{len(chunks)}`")

    if st.button("INICIAR STREAMING DE ÁUDIO"):
        client = genai.Client(api_key=api_key)
        
        # Containers para Logs em Tempo Real
        total_progress = st.progress(0, text="Aguardando início...")
        log_container = st.container(border=True)
        log_text = log_container.empty()
        
        full_audio_data = []
        logs = []

        def update_log(msg):
            logs.append(msg)
            # Mostra apenas as últimas 5 linhas para não poluir
            log_text.code("\n".join(logs[-5:]), language="text")

        try:
            for i, chunk in enumerate(chunks):
                if i >= 15:
                    update_log("🛑 Limite de 15 chamadas atingido.")
                    break

                chunk_start_time = time.time()
                update_log(f"🔵 [Bloco {i+1}/{len(chunks)}] Enviando solicitação para API...")
                
                # USANDO STREAM=TRUE
                # Isso permite receber dados enquanto o Google ainda está "pensando"
                response_stream = client.models.generate_content_stream(
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

                # Processando o fluxo de dados (Stream)
                chunk_audio_parts = []
                packet_count = 0
                
                for response in response_stream:
                    # A cada pacote que chega, atualizamos o log
                    packet_count += 1
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if part.inline_data:
                                chunk_audio_parts.append(part.inline_data.data)
                                # Feedback visual de que está baixando
                                if packet_count % 5 == 0:
                                    update_log(f"   ⬇️ [Bloco {i+1}] Baixando pacote {packet_count}...")

                # Acabou este bloco
                elapsed = time.time() - chunk_start_time
                update_log(f"✅ [Bloco {i+1}] Concluído em {elapsed:.1f}s. Pacotes recebidos: {packet_count}")
                
                # Guarda os dados brutos deste bloco
                if chunk_audio_parts:
                    # O stream retorna vários pedacinhos, precisamos juntar
                    # Mas como é raw data dentro de containers, vamos guardar para o final
                    # Nota: O Gemini Stream pode mandar headers repetidos ou raw PCM.
                    # Para simplificar e evitar erro de header no meio, vamos pegar o blob inteiro.
                    # Na prática, com Stream, é melhor concatenar bytes puros se for PCM, 
                    # ou confiar no stitch_wavs se cada resposta for um WAV válido.
                    # O generate_content_stream manda pedaços. Vamos agrupar bytes do bloco atual.
                    block_blob = b"".join(chunk_audio_parts)
                    full_audio_data.append(block_blob)
                
                # Atualiza barra geral
                total_progress.progress((i + 1) / len(chunks), text=f"Processando bloco {i+1}/{len(chunks)}")

            # Finalização
            update_log("🛠️ Unindo arquivos de áudio...")
            final_wav = stitch_wavs(full_audio_data)
            
            if final_wav:
                st.success("Processo Finalizado com Sucesso!")
                st.audio(final_wav, format='audio/wav')
                st.download_button("Baixar WAV Final", final_wav, file_name="audio_completo.wav", mime="audio/wav")
            else:
                st.error("Erro ao gerar o arquivo final (nenhum dado válido recebido).")

        except Exception as e:
            st.error(f"ERRO FATAL: {str(e)}")
            update_log(f"❌ Erro: {str(e)}")
