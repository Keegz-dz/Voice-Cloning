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
from scripts import audio_enhancing

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
        
        # Todo: @atharvchaudhary696 Add your custom preprocess function
        
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
                # audio_enhancing.main(orig_wav, sr,r"\enhanced_audio.wav")
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
        """Setup the output and analysis tab that showcases the strengths of the cloned voice."""

        if st.session_state.cloned_wav is None or not isinstance(st.session_state.cloned_wav, np.ndarray) or st.session_state.cloned_wav.size == 0:
            st.info("Generate a cloned voice first in the 'Source Audio and Cloning' tab.")
            return

        try:
            # Create a selectbox selector for different views instead of tabs/radio buttons
            view_option = st.selectbox(
                "Select View:",
                ["Cloned Voice", "Voice Characteristics"],
                key="output_view_selector"
            )

            # Display content based on selection
            if view_option == "Cloned Voice":
                self._setup_cloned_voice_section()
            elif view_option == "Voice Characteristics":
                self._setup_voice_characteristics_profile()

        except Exception as e:
            st.error(f"Error in analysis tab: {str(e)}")
            st.info("Try generating a new cloned voice or reset the application")

    def _setup_cloned_voice_section(self) -> None:
        """Setup the main cloned voice section with audio player and download options."""
        st.text("")
        st.text("")
        st.markdown("### Synthesized Content")
        st.text("")
        
        col1, col2 = st.columns(2)
        with col1:
            if "cloned_wav" in st.session_state and isinstance(st.session_state.cloned_wav, np.ndarray) and st.session_state.cloned_wav.size > 0 and "sr" in st.session_state:
                    try:
                        # Convert numpy array to bytes
                        byte_io = io.BytesIO()
                        sf.write(byte_io, st.session_state.cloned_wav, st.session_state.sr, format="WAV")
                        cloned_audio_bytes = byte_io.getvalue()
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
                
        # Add statistics about the generated audio
        if st.session_state.cloned_wav is not None:
            sr = st.session_state.sr
            duration = len(st.session_state.cloned_wav) / sr
            st.caption(f"{duration:.2f} seconds | {sr} Hz")
            
        # Display text used for synthesis
        st.info(st.session_state.text_input if st.session_state.text_input else "No text provided")
        

    def _setup_voice_characteristics_profile(self) -> None:
        """Display voice characteristics profile with metrics calculated from actual audio data."""
        
        try:
            st.markdown("### Voice Characteristics Profile")
            st.markdown("""
            The following analysis shows the key characteristics of your cloned voice based on advanced acoustic analysis.
            """)
            
            if hasattr(st.session_state, "cloned_wav") and st.session_state.cloned_wav is not None:
                # Calculate actual metrics from the cloned audio data
                from data_scripts.voice_metrics import calculate_voice_metrics
                
                with st.spinner("Analyzing voice characteristics..."):
                    voice_metrics = calculate_voice_metrics(st.session_state.cloned_wav, st.session_state.sr)
                    
                    if voice_metrics is None:
                        st.warning("Could not analyze voice characteristics from the audio data.")
                        return
                
                
                # Display the metrics in a radar chart
                categories = list(voice_metrics.keys())
                values = [voice_metrics[category] for category in categories]
                
                # Fill in the values for the radar chart
                values = np.array(values)
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                values = np.concatenate((values, [values[0]]))  # Close the loop
                angles = np.concatenate((angles, [angles[0]]))  # Close the loop
                categories = np.concatenate((categories, [categories[0]]))  # Close the loop
                
                # Create radar chart with more compact size
                fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
                ax.fill(angles, values, color='#5066D2', alpha=0.25)
                ax.plot(angles, values, 'o-', linewidth=2, color='#5066D2')
                
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1], fontsize=12)
                
                ax.set_ylim(0, 100)
                ax.set_yticks([20, 40, 60, 80, 100])
                ax.set_yticklabels(['20', '40', '60', '80', 'Excellent'])
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # # Save with high DPI
                # buf = io.BytesIO()
                # fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                # buf.seek(0)
                # # Display high-res image at desired size
                # st.image(buf, use_container_width=False, width=500)
                
                # Display voice characteristic interpretations in collapsible sections
                st.markdown("### Characteristic Details")
                
                # Create a 2x2 grid for metrics explanations with collapsible sections
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander(f"Timbre Richness: {voice_metrics['Timbre Richness']:.1f}/100", expanded=True):
                        st.markdown("""
                        - Refers to the voice's tonal quality and warmth
                        - Higher values indicate richer harmonic structure
                        - Rich overtones create a natural, warm vocal quality
                        """)
                        
                        # Add explanation based on the actual score
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
                        
                        # Add explanation based on the actual score
                        score = voice_metrics['Pitch Stability']
                        if score >= 90:
                            st.success("Excellent: The voice shows very stable pitch with natural micro-variations.")
                        elif score >= 75:
                            st.info("Good: The voice has good pitch stability with appropriate variations.")
                        else:
                            st.warning("Moderate: The pitch control could be more consistent.")
                    
                # Second row
                col3, col4 = st.columns(2)
                
                with col3:
                    with st.expander(f"Articulation: {voice_metrics['Articulation']:.1f}/100", expanded=True):
                        st.markdown("""
                        - Represents clarity of consonants and vowels
                        - Higher values indicate clearer pronunciation
                        - Well-defined phoneme boundaries create intelligible speech
                        """)
                        
                        # Add explanation based on the actual score
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
                        
                        # Add explanation based on the actual score
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