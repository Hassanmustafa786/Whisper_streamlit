import streamlit as st
import whisper
import os

# Load the Whisper model
model = whisper.load_model("base")

# Function to transcribe audio
def transcribe(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    st.write(f"Detected language: {detected_language}")
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

# Streamlit app
def main():
    st.title('Whisper')
    code = '''Made by Hassan Mustafa.'''
    st.code(code, language='python')

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file:
        
        # Save the uploaded file locally
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        st.audio("temp_audio.wav", format='audio/wav')

        if st.button('Transcribe'):
            transcription = transcribe("temp_audio.wav")
            st.subheader('Transcription:')
            st.text_area("Transcription Output", value=transcription, height=200)

        # Remove the temporary file
        os.remove("temp_audio.wav")

if __name__ == '__main__':
    main()