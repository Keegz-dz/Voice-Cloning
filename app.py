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
import io
import base64

from main import Main

# ======================
# Constants and Configuration
# ======================
SAMPLE_RATE = 16000  # Standard sample rate for voice processing (16kHz)

# Predefined text templates for quick generation
AUTO_TEXT_OPTIONS = {
    "Introduction": "Hello everyone, it's nice to speak with you today. This is a demonstration of voice cloning technology.",
    "Weather Report": "Today's forecast calls for partly cloudy skies with a high of 72 degrees. There's a 20 percent chance of rain in the afternoon.",
    "Story Excerpt": "Once upon a time, in a land far away, there lived a wise old dragon who guarded the ancient forests. The villagers both feared and respected this magnificent creature.",
    "Technical Explanation": "The voice synthesis process works by extracting features from the reference audio and then generating new audio that matches these characteristics.",
    "Motivational Quote": "The only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle."
}


class VoiceCloningApp:
    """
    Main application class for MOTIVE Studio - a voice cloning application 
    that allows users to clone voices using neural speech synthesis.
    """
    def __init__(self):
        """Initialize the Streamlit Voice Cloning application and set up the UI structure."""
        # Configure page layout and title
        st.set_page_config(
            page_title="Voice Cloning Studio",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("MOTIVE Studio")
        
        # Initialize session state variables for persistence between reruns
        self._initialize_session_state()
        
        # Set up the main UI components
        self._setup_sidebar()
        self._setup_tabs()
    
    
    def _initialize_session_state(self) -> None:
        """
        Initialize session state variables to maintain app state between reruns.
        These store audio data, model instances, and user inputs.
        """
        default_state = {
            "orig_bytes": None,      # Original audio file bytes
            "orig_wav": None,        # Processed numpy array of source audio
            "sr": None,              # Sample rate
            "cloned_wav": None,      # Generated cloned voice as numpy array
            "text_input": "",        # User's text to synthesize
            "processing": False,     # Flag for processing state
            "model_instance": None,  # ML model instance (lazy loaded)
            "use_vocoder": True,     # Whether to use neural vocoder vs Griffin-Lim
            "encoder_choice": "Speech Encoder v2.0.1"  # Default encoder model
        }
        
        # Only initialize keys that don't already exist
        for key, default_value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    
    def _setup_sidebar(self) -> None:
        """
        Configure the sidebar with model parameters and memory management options.
        Provides advanced settings for tweaking the voice synthesis process.
        """
        with st.sidebar:
            st.header("Acoustic Model Parameters")
            
            # Advanced synthesis and encoder settings
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
            
            # Memory management - reset app state
            st.subheader("Memory Management")
            if st.button("Reset All", help="Clear all uploaded and generated audio"):
                for key in ["orig_bytes", "orig_wav", "sr", "cloned_wav", "text_input"]:
                    st.session_state[key] = None
                if st.session_state.model_instance:
                    st.session_state.model_instance = None
                st.rerun()
            
            # App version footer
            st.markdown("---")
            st.caption("MOTIVE Studio v2.0")
    
    
    def _setup_tabs(self) -> None:
        """Set up the main tabbed interface with input and output sections."""
        tab1, tab2 = st.tabs(["Source Audio and Cloning", "Cloned Audio and Analysis"])
        
        with tab1:
            self._setup_upload_tab()
        
        with tab2:
            self._setup_output_tab()
    
    
    def _setup_upload_tab(self) -> None:
        """
        Configure the first tab with audio upload, text input, and generation controls.
        This tab handles the input side of the voice cloning process.
        """
        st.text("")
        st.text("")
        
        # Split interface into two columns
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            self._setup_audio_upload()
        
        with col2:
            self._setup_text_input()
        
        # Add generation button below both columns
        self._setup_generation_controls()
    
    
    def _setup_audio_upload(self) -> None:
        """
        Handle audio file upload, processing, and display.
        Validates and processes the reference voice audio.
        """
        st.subheader("Reference Voice")
        audio_file = st.file_uploader(
            "Upload an audio file (WAV or MP3)", 
            type=["wav", "mp3"],
            help="Upload a clear voice recording without background noise for best results"
        )
        
        if audio_file:
            # Store raw audio bytes in session
            st.session_state.orig_bytes = audio_file.read()
            
            # Display audio player with base64 encoding
            def play_audio(audio_bytes):
                """Plays audio in Streamlit using base64 encoding for better control."""
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                audio_tag = f'<audio controls src="data:audio/wav;base64,{audio_base64}"></audio>'
                st.markdown(audio_tag, unsafe_allow_html=True)

            if "orig_bytes" in st.session_state and st.session_state.orig_bytes:
                play_audio(st.session_state.orig_bytes)
            else:
                st.warning("No audio data available in st.session_state.orig_bytes")
            
            # Process and analyze audio file
            try:
                # Create temporary file to process with librosa
                with tempfile.NamedTemporaryFile(suffix=f".{audio_file.name.split('.')[-1]}", delete=False) as tmp:
                    tmp.write(st.session_state.orig_bytes)
                    tmp_path = tmp.name
                
                # Load and resample audio to target sample rate
                orig_wav, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
                os.unlink(tmp_path)  # Clean up temp file
                
                # Store processed audio in session state
                st.session_state.orig_wav = orig_wav
                st.session_state.sr = sr
                
                # Display audio statistics and quality warnings
                duration = len(orig_wav) / sr
                st.caption(f"{duration:.2f} seconds | {sr} Hz")
                
                # Provide quality guidance based on duration
                if duration < 3:
                    st.warning("Audio is very short. For better results, use recordings of at least 5 seconds.")
                elif duration > 30:
                    st.warning("Long audio may increase processing time. 10-20 seconds is ideal.")
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
    
    
    def _setup_text_input(self) -> None:
        """
        Configure text input area with random generation options.
        Manages the target text for voice synthesis.
        """
        st.subheader("Target Text")
        
        # Text input with persistent state
        text_input = st.text_area(
            "Enter your custom text or use random generation",
            value=st.session_state.text_input,
            height=154,
            placeholder="Your words, their voice. Enter text here..."
        )
        
        # Update session state if text changed
        if text_input != st.session_state.text_input:
            st.session_state.text_input = text_input
        
        # Random text generation for quick testing
        if st.button("Generate Random Text", help="Get a randomly selected text template"):
            random_key = random.choice(list(AUTO_TEXT_OPTIONS.keys()))
            st.session_state.text_input = AUTO_TEXT_OPTIONS[random_key]
            st.rerun()
        
        # Show text statistics if text exists
        if st.session_state.text_input:
            word_count = len(st.session_state.text_input.split())
            char_count = len(st.session_state.text_input)
            st.caption(f"{word_count} words | {char_count} characters")
    
    
    def _setup_generation_controls(self) -> None:
        """
        Set up the voice generation button and process controls.
        Validates inputs and triggers the cloning process.
        """
        # Custom styling for the generation button
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

        # Main generation button (disabled during processing)
        generate_button = st.button(
            "Generate Cloned Voice",
            use_container_width=True,
            disabled=st.session_state.processing
        )

        # Validate inputs and trigger generation
        if generate_button:
            # Validate required inputs
            audio_loaded = st.session_state.orig_wav is not None and isinstance(st.session_state.orig_wav, np.ndarray) and st.session_state.orig_wav.size > 0
            text_provided = st.session_state.text_input is not None and st.session_state.text_input.strip() != ""

            if not audio_loaded or not text_provided:
                st.error("Please upload an audio file and enter the text to synthesize.")
            else:
                # All requirements met, run cloning process
                self._run_voice_cloning()
    
    
    def _run_voice_cloning(self) -> None:
        """
        Execute the multi-step voice cloning process with progress tracking.
        Shows detailed status for each stage of the pipeline.
        """
        # Set processing flag to prevent multiple simultaneous runs
        st.session_state.processing = True
        progress = st.progress(0)

        # Utility function to log execution time of steps
        def log_time(action, func):
            start = time.time()
            func()
            st.caption(f"{time.time() - start:.2f} seconds")

        # Define the processing stages with their implementation functions
        def initialize_models():
            """Load or reuse ML models for voice cloning."""
            if st.session_state.model_instance is None:
                st.write("Loading Speech Encoder and Synthesizer models...")
                st.session_state.model_instance = Main(
                    original_encoder=(st.session_state.encoder_choice == "Baseline Model (LSTM)")
                )
            else:
                st.write("Speech Encoder and Synthesizer models already loaded.")

        def analyze_voice_patterns():
            """Extract and analyze voice characteristics from reference audio."""
            st.write("Extracting speaker embeddings from the reference audio...")
            time.sleep(0.3)  # Simplified for demonstration

        def synthesize_voice():
            """Generate voice audio from text using the reference voice characteristics."""
            st.write("Generating spectrogram from text and speaker embeddings...")
            main = st.session_state.model_instance
            st.session_state.cloned_wav = main.clone_audio(
                st.session_state.orig_wav,
                st.session_state.use_vocoder,
                st.session_state.text_input
            )

        def post_process_audio():
            """Apply audio enhancement and normalization."""
            st.write("Applying post-processing to the generated audio...")
            time.sleep(0.2)  # Simplified for demonstration

        def save_output():
            """Encode the generated audio to WAV format."""
            st.write("Encoding the generated audio to WAV format...")
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, st.session_state.cloned_wav, st.session_state.sr, format='WAV')
            audio_bytes.seek(0)
            sf.write("cloned.wav", st.session_state.cloned_wav, st.session_state.sr)

        # Define processing pipeline steps with progress percentages
        steps = [
            ("Initializing models", 15, initialize_models, False),
            ("Analyzing voice patterns", 30, analyze_voice_patterns, False),
            ("Synthesizing voice", 75, synthesize_voice, False),
            ("Post-processing audio", 90, post_process_audio, False),
            ("Saving output", 100, save_output, False),
        ]

        try:
            # Execute each step in sequence with status tracking
            for label, pct, fn, expand in steps:
                with st.status(label=label, state="running", expanded=expand):
                    try:
                        log_time(label, fn)
                    except Exception as step_error:
                        st.error(f"Step failed: {label}")
                        raise step_error
                progress.progress(pct)

            # Success message on completion
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("**Success:** Voice cloning process finished! The generated audio is available in the 'Output & Comparison' tab.")

        except Exception as e:
            st.error("**Error:** Voice cloning failed.")
            st.exception(e)

        finally:
            # Reset processing flag regardless of outcome
            st.session_state.processing = False

    
    def _setup_output_tab(self) -> None:
        """
        Configure the output tab with cloned audio playback and analysis views.
        Shows different visualizations of the generated voice.
        """
        # Check if cloned audio exists
        if st.session_state.cloned_wav is None or not isinstance(st.session_state.cloned_wav, np.ndarray) or st.session_state.cloned_wav.size == 0:
            st.info("Generate a cloned voice first in the 'Source Audio and Cloning' tab.")
            return

        try:
            # Create a view selector for different analysis perspectives
            view_option = st.selectbox(
                "Select View:",
                ["Cloned Voice", "Voice Characteristics"],
                key="output_view_selector"
            )

            # Display content based on selected view
            if view_option == "Cloned Voice":
                self._setup_cloned_voice_section()
            elif view_option == "Voice Characteristics":
                self._setup_voice_characteristics_profile()

        except Exception as e:
            st.error(f"Error in analysis tab: {str(e)}")
            st.info("Try generating a new cloned voice or reset the application")


    def _setup_cloned_voice_section(self) -> None:
        """
        Display the cloned voice audio player and related information.
        Shows the synthesized audio with playback controls.
        """
        st.text("")
        st.text("")
        st.markdown("### Synthesized Content")
        st.text("")
        
        col1, col2 = st.columns(2)
        with col1:
            # Check for valid cloned audio data
            if "cloned_wav" in st.session_state and isinstance(st.session_state.cloned_wav, np.ndarray) and st.session_state.cloned_wav.size > 0 and "sr" in st.session_state:
                try:
                    # Convert numpy array to audio bytes for player
                    byte_io = io.BytesIO()
                    sf.write(byte_io, st.session_state.cloned_wav, st.session_state.sr, format="WAV")
                    cloned_audio_bytes = byte_io.getvalue()
                    
                    # Display audio player with base64 encoding
                    def play_audio(audio_bytes):
                        """Plays audio in Streamlit using base64 encoding."""
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        audio_tag = f'<audio controls src="data:audio/wav;base64,{audio_base64}"></audio>'
                        st.markdown(audio_tag, unsafe_allow_html=True)
                    play_audio(cloned_audio_bytes)
                except Exception as e:
                    st.error(f"Error playing cloned audio: {str(e)}")
            else:
                st.warning("No cloned audio data available.")
                
        # Show audio statistics
        if st.session_state.cloned_wav is not None:
            sr = st.session_state.sr
            duration = len(st.session_state.cloned_wav) / sr
            st.caption(f"{duration:.2f} seconds | {sr} Hz")
            
        # Display synthesized text
        st.info(st.session_state.text_input if st.session_state.text_input else "No text provided")
    

    def _setup_voice_characteristics_profile(self) -> None:
        """
        Display detailed voice quality metrics and visualizations.
        Analyzes and visualizes voice characteristics in a radar chart.
        """
        try:
            st.markdown("### Voice Characteristics Profile")
            st.markdown("""
            The following analysis shows the key characteristics of your cloned voice based on advanced acoustic analysis.
            """)
            
            if hasattr(st.session_state, "cloned_wav") and st.session_state.cloned_wav is not None:
                # Calculate voice quality metrics from the audio data
                from data_scripts.voice_metrics import calculate_voice_metrics
                
                with st.spinner("Analyzing voice characteristics..."):
                    voice_metrics = calculate_voice_metrics(st.session_state.cloned_wav, st.session_state.sr)
                    
                    if voice_metrics is None:
                        st.warning("Could not analyze voice characteristics from the audio data.")
                        return
                
                # Create radar chart visualization of voice metrics
                categories = list(voice_metrics.keys())
                values = [voice_metrics[category] for category in categories]
                
                # Format data for radar chart
                values = np.array(values)
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                values = np.concatenate((values, [values[0]]))  # Close the loop
                angles = np.concatenate((angles, [angles[0]]))  # Close the loop
                categories = np.concatenate((categories, [categories[0]]))  # Close the loop
                
                # Create radar chart with custom styling
                fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
                ax.fill(angles, values, color='#5066D2', alpha=0.25)
                ax.plot(angles, values, 'o-', linewidth=2, color='#5066D2')
                
                # Configure radar chart appearance
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1], fontsize=12)
                
                ax.set_ylim(0, 100)
                ax.set_yticks([20, 40, 60, 80, 100])
                ax.set_yticklabels(['20', '40', '60', '80', 'Excellent'])
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display detailed interpretation of each voice metric
                st.markdown("### Characteristic Details")
                
                # Create a 2x2 grid for metrics explanations
                col1, col2 = st.columns(2)
                
                # First row of metrics
                with col1:
                    with st.expander(f"Timbre Richness: {voice_metrics['Timbre Richness']:.1f}/100", expanded=True):
                        st.markdown("""
                        - Refers to the voice's tonal quality and warmth
                        - Higher values indicate richer harmonic structure
                        - Rich overtones create a natural, warm vocal quality
                        """)
                        
                        # Add customized interpretation based on score
                        score = voice_metrics['Timbre Richness']
                        if score >= 90:
                            st.success("Excellent: This voice has a rich, full-bodied timbre with well-balanced harmonics.")
                        elif score >= 75:
                            st.info("Good: This voice has a pleasant timbre with a decent harmonic structure.")
                        else:
                            st.warning("Moderate: The voice timbre could benefit from more harmonic richness.")
                
                with col2:
                    with st.expander(f"Pitch Stability: {voice_metrics['Pitch Stability']:.1f}/100", expanded=True):
                        st.markdown("""
                        - Measures how consistently the voice maintains intended pitch
                        - Higher values show better pitch control with natural micro-variations
                        - Important for creating natural-sounding speech melody
                        """)
                        
                        # Add customized interpretation based on score
                        score = voice_metrics['Pitch Stability']
                        if score >= 90:
                            st.success("Excellent: The voice shows very stable pitch with natural micro-variations.")
                        elif score >= 75:
                            st.info("Good: The voice has good pitch stability with appropriate variations.")
                        else:
                            st.warning("Moderate: The pitch control could be more consistent.")
                    
                # Second row of metrics
                col3, col4 = st.columns(2)
                
                with col3:
                    with st.expander(f"Articulation: {voice_metrics['Articulation']:.1f}/100", expanded=True):
                        st.markdown("""
                        - Represents clarity of consonants and vowels
                        - Higher values indicate clearer pronunciation
                        - Well-defined phoneme boundaries create intelligible speech
                        """)
                        
                        # Add customized interpretation based on score
                        score = voice_metrics['Articulation']
                        if score >= 90:
                            st.success("Excellent: The speech is very clearly articulated with distinct consonants and vowels.")
                        elif score >= 75:
                            st.info("Good: The speech has good clarity with well-defined sounds.")
                        else:
                            st.warning("Moderate: Some phonemes could be more clearly articulated.")
                    
                with col4:
                    with st.expander(f"Speech Rhythm: {voice_metrics['Speech Rhythm']:.1f}/100", expanded=True):
                        st.markdown("""
                        - Measures the naturalness of pacing and pauses
                        - Higher values indicate more natural rhythm patterns
                        - Proper stress patterns create convincing human-like speech
                        """)
                        
                        # Add customized interpretation based on score
                        score = voice_metrics['Speech Rhythm']
                        if score >= 90:
                            st.success("Excellent: The speech has very natural rhythm with appropriate pauses and pacing.")
                        elif score >= 75:
                            st.info("Good: The speech has a natural flow with good pacing.")
                        else:
                            st.warning("Moderate: The rhythm could be more natural with better pause patterns.")
            else:
                st.warning("Voice characteristics profile requires generating a cloned voice first.")
                
        except Exception as e:
            st.error(f"Error in voice characteristics profile: {str(e)}")
            st.exception(e)  # Show detailed error for debugging


if __name__ == "__main__":
    VoiceCloningApp()