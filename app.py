import streamlit as st
import os
from main_model import main
import tempfile

# Set page config without icons or emojis
st.set_page_config(
    page_title="Voice Synthesis App",
    layout="wide"
)

# Enhanced custom CSS for a polished, user-friendly UI
st.markdown("""
    <style>
    /* Overall background and font styles */
    body {
        background-color: #f7f7f7;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        color: #333;
    }
    
    /* Title styling */
    .title {
        font-size: 3em;
        font-weight: 700;
        text-align: center;
        margin: 20px 0 40px 0;
        color: #2c3e50;
    }
    
    /* Sidebar styling */
    .sidebar .css-1d391kg {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    /* Header styling in sidebar */
    .sidebar .css-1d391kg h1, .sidebar .css-1d391kg h2, .sidebar .css-1d391kg h3 {
        color: #2c3e50;
    }
    
    /* Input section styling */
    .stTextArea, .stFileUploader {
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2980b9;
        color: white;
        padding: 12px 28px;
        border-radius: 5px;
        border: none;
        font-size: 1em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1f6391;
    }
    
    /* Audio player styling */
    .stAudio {
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Download button styling */
    .stDownloadButton button {
        background-color: #27ae60;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 1em;
    }
    .stDownloadButton button:hover {
        background-color: #1e8449;
    }
    
    /* Output section styling */
    .output-section {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.markdown('<p class="title">Voice Synthesis App</p>', unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.header("About")
    st.write("""
    This application synthesizes speech in a voice similar to your provided reference audio.
    Upload your audio file and enter the text to synthesize. The app will generate and play the resulting voice.
    """)
    st.markdown("---")
    st.write("**Instructions:**")
    st.write("1. Upload a reference audio file (WAV or FLAC)")
    st.write("2. Enter the text you want to synthesize")
    st.write("3. Click 'Synthesize Voice'")
    st.write("4. Listen and download the generated audio")
    st.markdown("---")
    st.write("Note: Processing may take a few moments depending on the text length.")

# Main content layout with two columns
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("Input")
    
    # File uploader for the reference audio file
    uploaded_file = st.file_uploader(
        "Upload reference audio (WAV/FLAC)", 
        type=["wav", "flac"],
        help="Upload a clear audio sample of the voice you want to mimic"
    )
    
    # Text input area for synthesis
    input_text = st.text_area(
        "Text to synthesize",
        height=200,
        value="I went to the zoo with my family. We saw birds, lions, and other fascinating animals. It was a wonderful day.",
        help="Enter the text you want the synthesized voice to say"
    )

with col2:
    st.header("Output")
    st.markdown('<div class="output-section">', unsafe_allow_html=True)
    if uploaded_file and input_text:
        if st.button("Synthesize Voice", use_container_width=True):
            with st.spinner("Processing... This may take a few moments..."):
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".flac") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                try:
                    # Run the synthesis process
                    main(audio_path, input_text)
                    
                    # Notify user on success
                    st.success("Synthesis complete!")
                    
                    # Display audio player for the generated output
                    st.audio("vocoder_output.wav")
                    
                    # Download button for the generated audio file
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
                    # Remove temporary file after processing
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
    else:
        st.info("Please upload an audio file and enter the text you want to synthesize.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.write("Voice Synthesis App - Powered by Streamlit")
