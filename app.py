import streamlit as st
import os
from main_model import main
import tempfile

# Set page config
st.set_page_config(
    page_title="Voice Synthesis App",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader>div>div>button {
        background-color: #2196F3;
        color: white;
    }
    .stAudio {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 2.5em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.markdown('<p class="title">Voice Synthesis App</p>', unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write("""
    This app allows you to synthesize speech in a voice similar to your reference audio.
    Upload an audio file and enter the text you want to synthesize, then click the button to generate.
    """)
    st.markdown("---")
    st.write("**Instructions:**")
    st.write("1. Upload a reference audio file (WAV or FLAC)")
    st.write("2. Enter the text you want to synthesize")
    st.write("3. Click 'Synthesize Voice'")
    st.write("4. Play the generated audio")
    st.markdown("---")
    st.write("Note: Processing may take a few moments depending on the length of the text.")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload reference audio (WAV/FLAC)", 
        type=["wav", "flac"],
        help="Upload a clear audio sample of the voice you want to mimic"
    )
    
    # Text input
    input_text = st.text_area(
        "Text to synthesize",
        height=200,
        value="I went to the zoo with my family, we saw birds, lions and various other animals. We enjoyed the day.",
        help="Enter the text you want the voice to say"
    )

with col2:
    st.header("Output")
    
    if uploaded_file and input_text:
        if st.button("Synthesize Voice", use_container_width=True):
            with st.spinner("Processing... This may take a few moments..."):
                # Save uploaded file to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                try:
                    # Run the synthesis
                    main(audio_path, input_text)
                    
                    # Display the result
                    st.success("Synthesis complete!")
                    
                    # Play the generated audio
                    st.audio("vocoder_output.wav")
                    
                    # Download button
                    with open("vocoder_output.wav", "rb") as f:
                        st.download_button(
                            label="Download Audio",
                            data=f,
                            file_name="synthesized_voice.wav",
                            mime="audio/wav"
                        )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
    else:
        st.info("Please upload an audio file and enter text to synthesize")

# Footer
st.markdown("---")
st.write("Voice Synthesis App - Powered by Streamlit")