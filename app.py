import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import time
import random
import threading
from typing import Tuple, Optional, Dict, Any
import io
import base64
from python_speech_features import mfcc
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from main import Main

# ======================
# Constants
# ======================
SAMPLE_RATE = 16000
AUTO_TEXT_OPTIONS = {
    "Introduction": "Hello everyone, it's nice to speak with you today. This is a demonstration of voice cloning technology.",
    "Weather Report": "Today's forecast calls for partly cloudy skies with a high of 72 degrees. There's a 20 percent chance of rain in the afternoon.",
    "Story Excerpt": "Once upon a time, in a land far away, there lived a wise old dragon who guarded the ancient forests. The villagers both feared and respected this magnificent creature.",
    "Technical Explanation": "The voice synthesis process works by extracting features from the reference audio and then generating new audio that matches these characteristics.",
    "Motivational Quote": "The only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle."
}


class VoiceCloningApp:
    def __init__(self):
        """Initialize the Streamlit Voice Cloning application."""
        # Page configuration
        st.set_page_config(
            page_title="Voice Cloning Studio",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("MOTIVE Studio")
        
        # Initialize session state variables
        self._initialize_session_state()
        
        # Display app components
        self._setup_sidebar()
        self._setup_tabs()
    
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables if they don't exist."""
        default_state = {
            "orig_bytes": None,
            "orig_wav": None,
            "sr": None,
            "cloned_wav": None,
            "text_input": "",
            "processing": False,
            "model_instance": None,
            "use_vocoder": True,
            "encoder_choice": "Speech Encoder v2.0.1"
        }
        
        for key, default_value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    
    def _setup_sidebar(self) -> None:
        """Setup the sidebar with settings and controls."""
        with st.sidebar:
            st.header("Acoustic Model Parameters")
            
            # Advanced settings in an expander
            with st.expander("Advanced Settings", expanded=False):
                st.text("")
                synthesis_method = st.selectbox(
                    "Choose the Audio Output Engine",
                    ("Neural Vocoder", "Griffin-Lim Algorithm"),
                    index=0,  # Default to Vocoder
                    help="Choose the method for converting spectrograms to audio."
                )
                st.session_state.use_vocoder = (synthesis_method == "Neural Vocoder")
                
                st.session_state.encoder_choice = st.selectbox(
                "Encoder Version",
                ("Speech Encoder v2.0.1", "Baseline Model (LSTM)"),
                help="The encoder utilizing our custom Transformer architecture, featuring Attention-Enhanced Speaker Representation with Contrastive Learning, generally demonstrates enhanced speaker discrimination capabilities compared to the LSTM encoder."
            )
            
            # Memory management
            st.subheader("Memory Management")
            if st.button("Reset All", help="Clear all uploaded and generated audio"):
                for key in ["orig_bytes", "orig_wav", "sr", "cloned_wav", "text_input"]:
                    st.session_state[key] = None
                if st.session_state.model_instance:
                    st.session_state.model_instance = None
                st.rerun()
            
            # Add a small footer
            st.markdown("---")
            st.caption("MOTIVE Studio v2.0")
            
    
    def _setup_tabs(self) -> None:
        """Setup the main tabs interface."""
        
        tab1, tab2 = st.tabs(["Source Audio and Cloning", "Cloned Audio and Analysis"])
        
        with tab1:
            self._setup_upload_tab()
        
        with tab2:
            self._setup_output_tab()
    
    
    def _setup_upload_tab(self) -> None:
        """Setup the upload and generation tab."""
        # st.header("Upload Reference Audio and Target Text")
        st.text("")
        st.text("")
        
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            self._setup_audio_upload()
        
        with col2:
            self._setup_text_input()
        
        self._setup_generation_controls()
    
    def _setup_audio_upload(self) -> None:
        """Setup the audio upload section."""
        st.subheader("Reference Voice")
        audio_file = st.file_uploader(
            "Upload an audio file (WAV or MP3)", 
            type=["wav", "mp3"],
            help="Upload a clear voice recording without background noise for best results"
        )
        
        if audio_file:
            # Store audio in session
            st.session_state.orig_bytes = audio_file.read()
            
            # Display audio player
            # st.audio(st.session_state.orig_bytes)
            def play_audio(audio_bytes):
                """Plays audio in Streamlit using base64 encoding."""
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_tag = f'<audio controls src="data:audio/wav;base64,{audio_base64}"></audio>'
                st.markdown(audio_tag, unsafe_allow_html=True)

            if "orig_bytes" in st.session_state and st.session_state.orig_bytes:
                play_audio(st.session_state.orig_bytes)
            else:
                st.warning("No audio data available in st.session_state.orig_bytes")
            
            # Show audio file info
            try:
                with tempfile.NamedTemporaryFile(suffix=f".{audio_file.name.split('.')[-1]}", delete=False) as tmp:
                    tmp.write(st.session_state.orig_bytes)
                    tmp_path = tmp.name
                
                orig_wav, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
                os.unlink(tmp_path)
                
                st.session_state.orig_wav = orig_wav
                st.session_state.sr = sr
                
                # Show audio details
                duration = len(orig_wav) / sr
                st.caption(f"{duration:.2f} seconds | {sr} Hz")
                
                if duration < 3:
                    st.warning("Audio is very short. For better results, use recordings of at least 5 seconds.")
                elif duration > 30:
                    st.warning("Long audio may increase processing time. 10-20 seconds is ideal.")
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
    
    
    def _setup_text_input(self) -> None:
        """Setup the text input section with random generation option."""
        st.subheader("Target Text")
        
        # Text input area
        text_input = st.text_area(
            "Enter your custom text or use random generation",
            value=st.session_state.text_input,
            height=154,
            placeholder="Your words, their voice. Enter text here..."
        )
        
        # Store text in session state
        if text_input != st.session_state.text_input:
            st.session_state.text_input = text_input
        
        # Random text generation
        if st.button("Generate Random Text",help="Get a randomly selected text template"):
            random_key = random.choice(list(AUTO_TEXT_OPTIONS.keys()))
            st.session_state.text_input = AUTO_TEXT_OPTIONS[random_key]
            st.rerun()
        
        # Text statistics
        if st.session_state.text_input:
            word_count = len(st.session_state.text_input.split())
            char_count = len(st.session_state.text_input)
            st.caption(f"{word_count} words | {char_count} characters")
    
    def _setup_generation_controls(self) -> None:
        """Setup the voice generation controls."""
        # Generation button
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: #007bff; /* Blue background */
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            }

            div.stButton > button:first-child:hover {
                background-color: #0056b3; /* Darker blue on hover */
            }

            div.stButton > button:first-child:disabled {
                background-color: #cccccc; /* Gray when disabled */
                color: #666666;
                cursor: not-allowed;
                box-shadow: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        generate_button = st.button(
            "Generate Cloned Voice",
            use_container_width=True,
            disabled=st.session_state.processing
        )

        # Validation and generation process
        if generate_button:
            # Check if audio is loaded and text is provided
            audio_loaded = st.session_state.orig_wav is not None and isinstance(st.session_state.orig_wav, np.ndarray) and st.session_state.orig_wav.size > 0
            text_provided = st.session_state.text_input is not None and st.session_state.text_input.strip() != ""

            if not audio_loaded or not text_provided:
                st.error("Please upload an audio file and enter the text to synthesize.")
            else:
                self._run_voice_cloning()
    
    def _run_voice_cloning(self) -> None:
        """Run the voice cloning process with progress tracking and status container per stage."""
        st.session_state.processing = True
        progress = st.progress(0)

        def log_time(action, func):
            start = time.time()
            func()
            st.caption(f"{time.time() - start:.2f} seconds")

        def initialize_models():
            if st.session_state.model_instance is None:
                st.write("Loading Speech Encoder and Synthesizer models...")
                st.session_state.model_instance = Main(
                    original_encoder=(st.session_state.encoder_choice == "Baseline Model (LSTM)")
                )
            else:
                st.write("Speech Encoder and Synthesizer models already loaded.")

        def analyze_voice_patterns():
            st.write("Extracting speaker embeddings from the reference audio...")
            time.sleep(0.3)

        def synthesize_voice():
            st.write("Generating spectrogram from text and speaker embeddings...")
            main = st.session_state.model_instance
            st.session_state.cloned_wav = main.clone_audio(
                st.session_state.orig_wav,
                st.session_state.use_vocoder,
                st.session_state.text_input
            )

        def post_process_audio():
            st.write("Applying post-processing to the generated audio...")
            time.sleep(0.2)

        def save_output():
            st.write("Encoding the generated audio to WAV format...")
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, st.session_state.cloned_wav, st.session_state.sr, format='WAV')
            audio_bytes.seek(0)
            sf.write("cloned.wav", st.session_state.cloned_wav, st.session_state.sr)

        steps = [
            ("Initializing models", 15, initialize_models, False),
            ("Analyzing voice patterns", 30, analyze_voice_patterns, False),
            ("Synthesizing voice", 75, synthesize_voice, False),
            ("Post-processing audio", 90, post_process_audio, False),
            ("Saving output", 100, save_output, False),
        ]

        try:
            for label, pct, fn, expand in steps:
                with st.status(label=label, state="running", expanded=expand):
                    try:
                        log_time(label, fn)
                    except Exception as step_error:
                        st.error(f"Step failed: {label}")
                        raise step_error
                progress.progress(pct)

            st.markdown("<br>", unsafe_allow_html=True)
            st.success("**Success:** Voice cloning process finished! The generated audio is available in the 'Output & Comparison' tab.")

        except Exception as e:
            st.error("**Error:** Voice cloning failed.")
            st.exception(e)

        finally:
            st.session_state.processing = False

    
    def _setup_output_tab(self) -> None:
        """Setup the output and comparison tab."""
        st.header("Output & Spectrogram Comparison")
        
        if st.session_state.cloned_wav is None or not isinstance(st.session_state.cloned_wav, np.ndarray) or st.session_state.cloned_wav.size == 0:
            st.info("Before you can proceed, please synthesize the speech in the cloned voice.")
            return
        
        try:
            # Results section
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.subheader("Cloned Voice Output")
                
                # Check if the file exists
                if os.path.exists("cloned.wav"):
                    st.audio("cloned.wav", format="audio/wav")
                    
                    # Create download button if file exists
                    try:
                        with open("cloned.wav", "rb") as f:
                            st.download_button(
                                "⬇️ Download Cloned Audio",
                                f,
                                "cloned_voice.wav",
                                "audio/wav",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error creating download button: {str(e)}")
                else:
                    # Generate file if it doesn't exist
                    try:
                        sf.write("cloned.wav", st.session_state.cloned_wav, st.session_state.sr)
                        st.audio("cloned.wav", format="audio/wav")
                        
                        with open("cloned.wav", "rb") as f:
                            st.download_button(
                                "⬇️ Download Cloned Audio",
                                f,
                                "cloned_voice.wav",
                                "audio/wav",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error creating audio file: {str(e)}")
                
                # Display text used for synthesis
                st.subheader("Text Used")
                st.info(st.session_state.text_input if st.session_state.text_input else "No text provided")
            
            with col2:
                # Wave comparison
                st.subheader("Waveform Comparison")
                self._plot_waveform_comparison()
            
            # Spectrogram comparison
            st.subheader("Spectrogram Comparison")
            
            with st.expander("Why Spectrograms Matter", expanded=False):
                st.markdown(
                    """
                    A spectrogram shows how frequency content changes over time. 
                    By comparing the original and cloned audio visually, you can see:
                    
                    - **Frequency patterns**: How closely the voice pitch matches
                    - **Temporal dynamics**: Speech rhythm and timing similarities
                    - **Harmonic structure**: Voice timbre and quality comparison
                    
                    The more similar these patterns appear, the better the voice cloning quality.
                    """
                )
            
            self._plot_spectrograms()
            
        except Exception as e:
            st.error(f"Error setting up output tab: {str(e)}")
            st.info("Try generating a new cloned voice or reset the application")
    
    def _plot_waveform_comparison(self) -> None:
        """Plot waveform comparison between original and cloned audio."""
        try:
            orig = st.session_state.orig_wav
            clone = st.session_state.cloned_wav
            sr = st.session_state.sr
            
            if orig is None or clone is None or sr is None:
                st.warning("Missing data for waveform comparison")
                return
                
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            # Plot original audio waveform
            librosa.display.waveshow(orig, sr=sr, ax=ax[0], color='#1f77b4')
            ax[0].set_title('Original Audio Waveform')
            ax[0].set_ylabel('Amplitude')
            ax[0].grid(True, alpha=0.3)
            
            # Plot cloned audio waveform
            librosa.display.waveshow(clone, sr=sr, ax=ax[1], color='#ff7f0e')
            ax[1].set_title('Cloned Audio Waveform')
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Amplitude')
            ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating waveform comparison: {str(e)}")
            # Provide a placeholder image
            st.image("https://via.placeholder.com/800x400?text=Waveform+Comparison+Error", use_column_width=True)
    
    def _plot_spectrograms(self) -> None:
        """Plot spectrograms of original and cloned audio for comparison."""
        try:
            orig = st.session_state.orig_wav
            clone = st.session_state.cloned_wav
            sr = st.session_state.sr
            
            if orig is None or clone is None or sr is None:
                st.warning("Missing data for spectrogram comparison")
                return
                
            # Compute spectrograms
            D1 = librosa.stft(orig)
            S1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
            D2 = librosa.stft(clone)
            S2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)
            
            # Plot spectrograms
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Spectrogram**")
                fig, ax = plt.subplots(figsize=(10, 5))
                img = librosa.display.specshow(
                    S1, 
                    sr=sr, 
                    x_axis="time", 
                    y_axis="log", 
                    ax=ax, 
                    cmap='viridis'
                )
                ax.set_title("Original Voice Spectrogram")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Cloned Spectrogram**")
                fig, ax = plt.subplots(figsize=(10, 5))
                img = librosa.display.specshow(
                    S2, 
                    sr=sr, 
                    x_axis="time", 
                    y_axis="log", 
                    ax=ax,
                    cmap='viridis'
                )
                ax.set_title("Cloned Voice Spectrogram")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating spectrograms: {str(e)}")

if __name__ == "__main__":
    VoiceCloningApp()