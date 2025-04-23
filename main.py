import torch
import librosa
import warnings
import logging
import numpy as np
from pathlib import Path

import params as p
from temp.audio import preprocess_wav
from embed import Embed
from vocoder import Vocoder
from utils.tacotron import Tacotron
from synthesizer import Synthesizer
from speech_encoder import SpeechEncoder
from speech_encoder_v2_updated import SpeechEncoderV2
from data_preprocessing import audio_preprocessing
from speech_2_text import SpeechTranslationPipeline


class Main():
    """
    Main controller class for the voice cloning system.
    
    This class orchestrates the three main components of the voice cloning pipeline:
    1. Encoder: Extracts speaker identity from reference audio
    2. Synthesizer: Generates mel spectrograms from text using speaker embeddings
    3. Vocoder: Converts spectrograms to time-domain waveforms
    """
    def __init__(self, original_encoder=False):
        """
        Initialize the voice cloning system with models.
        
        Args:
            original_encoder: If True, uses the baseline LSTM encoder model.
                             If False, uses the improved transformer-based encoder.
        """
        # Set up logging
        self.logger = self._setup_logger()
        
        # Suppress non-critical warnings
        warnings.filterwarnings("ignore")

        # Set up computation devices (GPU if available, otherwise CPU)
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.loss_device = torch.device("cpu")
            self.logger.info(f"Using device: {self.device}")
        except Exception as e:
            self.logger.error(f"Error in setting device: {e}")
            self.logger.error("Please check your CUDA installation.")
            # Fallback to CPU
            self.device = torch.device("cpu")
            self.loss_device = torch.device("cpu")
        
        # Initialize the three core neural models
        self.encoder = self.__init__encoder(original_encoder)
        self.synthesizer = self.__init__synthesizer()
        self.vocoder = self.__init__vocoder()

        # Create the embedding utility for voice identity extraction
        try:
            self.embedder = Embed(self.encoder)
            self.logger.info("Successfully initialized embedder.")
        except Exception as e:
            self.logger.error(f"Error in initializing embedder: {e}")
            self.logger.error("Please read the documentations incase of errors.")
            self.embedder = None

    def _setup_logger(self):
        """Set up and return a configured logger."""
        logger = logging.getLogger("VoiceCloning")
        logger.setLevel(logging.INFO)
        
        # Only add handler if not already set up
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def __init__encoder(self, original_encoder, 
                      encoder_path: str = "models/speech_encoder_transformer_updated/encoder_073500_loss_0.0724.pt"):
        """
        Initialize the speaker encoder model that extracts voice identity.
        
        The encoder converts reference audio into a fixed-dimensional embedding vector
        that captures the unique characteristics of the speaker's voice.
        
        Args:
            original_encoder: Boolean flag to select encoder architecture
            encoder_path: Path to the transformer encoder model weights
            
        Returns:
            Initialized encoder model
        """
        try:
            if original_encoder:
                # Initialize legacy LSTM-based encoder
                self.logger.info("Loading legacy LSTM encoder model...")
                encoder = SpeechEncoder(self.device, self.loss_device)
                checkpoints = torch.load(
                    "models/speech_encoder_lstm/encoder.pt",
                    map_location=self.device
                )
                encoder.load_state_dict(checkpoints['model_state'])
                self.logger.info("Successfully loaded the LSTM encoder model.")
                return encoder

            # Initialize improved transformer-based encoder with attention mechanism
            self.logger.info("Loading transformer encoder model...")
            encoder = SpeechEncoderV2(self.device, self.device)
            checkpoints = torch.load(
                encoder_path,
                map_location=self.device
            )
            encoder.load_state_dict(checkpoints['model_state'])
            self.logger.info("Successfully loaded the speaker encoder model.")
            return encoder

        except FileNotFoundError as e:
            self.logger.error(f"Encoder model file not found: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in loading encoder: {e}")
            self.logger.error("Please check the encoder model path.")
            return None
    
    def __init__synthesizer(self, synthesizer_path: Path = Path("models/synthesizer/synthesizer.pt")):
        """
        Initialize the synthesizer model that generates spectrograms.
        
        The synthesizer takes text and speaker embeddings as input and generates
        mel-spectrograms that represent the target speech with the voice characteristics
        from the embedding.
        
        Args:
            synthesizer_path: Path to the synthesizer model weights
            
        Returns:
            Initialized synthesizer model
        """
        try:
            self.logger.info("Loading synthesizer model...")
            synthesizer = Synthesizer(synthesizer_path)
            synthesizer.load()
            self.logger.info("Successfully loaded the synthesizer model.")
            return synthesizer
        except FileNotFoundError as e:
            self.logger.error(f"Synthesizer model file not found: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in loading synthesizer: {e}")
            self.logger.error("Please check the synthesizer model path.")
            return None

    def __init__vocoder(self, vocoder_path: str = "models/vocoder/vocoder.pt"):
        """
        Initialize the vocoder model for high-quality waveform generation.
        
        The vocoder converts spectrograms into high-quality audio waveforms.
        It produces more natural-sounding results than the Griffin-Lim algorithm.
        
        Args:
            vocoder_path: Path to the vocoder model weights
            
        Returns:
            Initialized vocoder model
        """
        try:
            self.logger.info("Loading vocoder model...")
            vocoder = Vocoder()
            vocoder.load_model(vocoder_path)
            self.logger.info("Successfully loaded the vocoder model.")
            return vocoder
        except FileNotFoundError as e:
            self.logger.error(f"Vocoder model file not found: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error in loading vocoder: {e}")
            self.logger.error("Please check the vocoder model path.")
            return None

    def clone_audio(self, audio, use_vocoder: bool = False, text = None):
        """
        Execute the complete voice cloning pipeline on input audio and text.
        
        This is the main function that performs the entire cloning process:
        1. Preprocess audio to standard format
        2. Extract text (transcribe or use provided)
        3. Extract speaker embedding (voice identity)
        4. Synthesize spectrograms with the voice identity
        5. Generate audio waveform (via vocoder or Griffin-Lim)
        
        Args:
            audio: Input audio as numpy array
            use_vocoder: Whether to use neural vocoder (higher quality but slower)
            text: Text to synthesize (optional); will transcribe from audio if None
            
        Returns:
            Synthesized audio as numpy array
        """
        self.logger.info("Model Initializations Completed.")  
        self.logger.info("Starting audio generation...")
        
        # Check if required models are initialized
        if not all([self.encoder, self.synthesizer, self.vocoder, self.embedder]):
            self.logger.error("Cannot proceed - one or more models failed to initialize")
            return np.zeros(16000)  # Return 1 second of silence
        
        # Step 1: Preprocess the input audio
        try:
            self.wav = preprocess_wav(audio, p.sample_rate)
            self.logger.info("Audio preprocessing completed.")
        except Exception as e:
            self.logger.error(f"Error in audio preprocessing: {e}")
            self.logger.error("Please provide a valid audio file.")
            return np.zeros(16000)  # Return 1 second of silence

        # Step 2: Get text to synthesize (either from parameter or via STT)
        try:
            if text is not None:
                self.logger.info("Using provided text for synthesis")
                # Split text into sentences using proper punctuation rules
                sentences = self._split_into_sentences(text)
                self.text = sentences
            else:
                self.logger.info("Transcribing audio to get text for synthesis")
                stt_model = SpeechTranslationPipeline()
                self.text = stt_model.transcribe_audio(self.wav).split("\n")
                
            # Filter empty entries
            self.text = [t.strip() for t in self.text if t and t.strip()]
            
            if not self.text:
                self.logger.warning("No valid text found. Using default text.")
                self.text = ["Error in processing text."]
        except Exception as e:
            self.logger.error(f"Error in speech-to-text: {e}")
            self.logger.error("Please check the audio file or the STT model.")
            # Fallback to prevent empty text
            if not hasattr(self, 'text') or not self.text:
                self.text = ["Error in processing text."]

        # Step 3: Extract speaker embedding (voice identity)
        try:
            embedding, partial_embeds, _ = self.embedder.embed_utterance(self.wav, return_partials=True)
            
            # Ensure we have enough embeddings for all sentences
            embeddings = [embedding] * len(self.text)
            
            self.logger.info(f"Processing {len(self.text)} sentence(s): {self.text}")
        except Exception as e:
            self.logger.error(f"Error in embedding: {e}")
            self.logger.error("Please check the audio file or the Embed model")
            return np.zeros(16000)  # Return 1 second of silence
        
        # Step 4: Generate mel spectrograms from text with speaker embedding
        try:
            # Filter out any empty or None text entries
            valid_text_entries = [t for t in self.text if t and isinstance(t, str) and t.strip()]
            if not valid_text_entries:
                valid_text_entries = ["No valid text to synthesize."]
                
            # Adjust embeddings to match valid text entries
            valid_embeddings = embeddings[:len(valid_text_entries)]
            if len(valid_embeddings) < len(valid_text_entries):
                valid_embeddings.extend([embedding] * (len(valid_text_entries) - len(valid_embeddings)))
            
            self.logger.info(f"Synthesizing {len(valid_text_entries)} valid sentence(s)")
            
            # Generate spectrograms for each text segment
            specs = self.synthesizer.synthesize_spectrograms(valid_text_entries, valid_embeddings)
            
            # Verify that specs is not empty before continuing
            if not specs or len(specs) == 0:
                self.logger.warning("No spectrograms were generated. Using fallback.")
                # Create a simple silence spectrogram as fallback
                from synthesizer.inference import Synthesizer as SynthesizerClass
                dummy_spec = np.zeros((SynthesizerClass.params.n_mels, 100))
                specs = [dummy_spec]
            
            # Concatenate all spectrograms
            spec = np.concatenate(specs, axis=1)
            breaks = [spec.shape[1] for spec in specs]
            
            self.logger.info(f"Generated {len(specs)} spectrograms with shapes: {[s.shape for s in specs]}")
        except Exception as e:
            self.logger.error(f"Error in synthesizer: {e}")
            self.logger.error("Error in generating spectrograms, refer to the documentation for more details.")
            self.logger.error("Using fallback spectrogram generation")
            from synthesizer.inference import Synthesizer as SynthesizerClass
            dummy_spec = np.zeros((SynthesizerClass.params.n_mels, 100))
            spec = dummy_spec
            breaks = [spec.shape[1]]

        # Step 5: Convert spectrograms to audio
        try:
            # First generate with Griffin-Lim algorithm (faster but lower quality)
            wav = self.synthesizer.griffin_lim(spec)
            wav = self.add_breaks(breaks, wav)

            # If requested, use neural vocoder for higher quality (slower)
            if use_vocoder:
                try:
                    self.logger.info("Using neural vocoder for higher quality audio...")
                    vocoder_wav = self.vocoder.infer_waveform(spec)
                    if vocoder_wav is not None and len(vocoder_wav) > 0:
                        wav = vocoder_wav
                        wav = self.add_breaks(breaks, wav)
                        self.logger.info("Vocoder audio generation successful.")
                    else:
                        self.logger.warning("Vocoder returned empty audio. Using Griffin-Lim output.")
                except Exception as vocoder_error:
                    self.logger.error(f"Vocoder error: {vocoder_error}. Falling back to Griffin-Lim output.")
                
            self.done()
            return wav
        
        except Exception as e:
            self.logger.error(f"Error in decoding: {e}")
            self.logger.error("Error occurred while decoding the spectrograms to audio. Please refer to the documentation for more details.")
            self.logger.error("Returning silent audio due to decoding error")
            return np.zeros(16000)  # 1 second of silence at 16kHz
        
    def _split_into_sentences(self, text):
        """
        Split text into sentences more reliably than simple newline splitting.
        
        Args:
            text: Input text to split into sentences
            
        Returns:
            List of sentences
        """
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split sentences on common ending punctuation followed by space and capital letter
        # or on newlines, with specified maximum length
        MAX_SENTENCE_LENGTH = 150  # Set maximum sentence length
        
        # Initial split on common sentence endings
        sentences = re.split(r'([.!?])\s+(?=[A-Z])', text)
        
        # Recombine the split punctuation marks
        processed_sentences = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                processed_sentences.append(sentences[i] + sentences[i+1])
            else:
                processed_sentences.append(sentences[i])
        
        # Further split any sentences that are too long
        final_sentences = []
        for sentence in processed_sentences:
            if len(sentence) > MAX_SENTENCE_LENGTH:
                # Split on commas, semicolons, or other logical breaks
                subsections = re.split(r'([,;:])\s+', sentence)
                
                # Recombine with punctuation
                current_section = ""
                for j in range(0, len(subsections)):
                    if j % 2 == 0:  # Text part
                        if current_section and len(current_section) + len(subsections[j]) > MAX_SENTENCE_LENGTH:
                            final_sentences.append(current_section.strip())
                            current_section = subsections[j]
                        else:
                            current_section += subsections[j]
                    else:  # Punctuation part
                        current_section += subsections[j] + " "
                
                if current_section:
                    final_sentences.append(current_section.strip())
            else:
                final_sentences.append(sentence.strip())
        
        # Filter out empty sentences
        final_sentences = [s for s in final_sentences if s.strip()]
        
        # If no sentences were found, return the original text as a single sentence
        if not final_sentences:
            return [text]
            
        return final_sentences

    def add_breaks(self, breaks, wav):
        """
        Add natural pauses between sentences in the generated audio.
        
        Args:
            breaks: List of spectrogram segment lengths
            wav: Audio waveform without breaks
            
        Returns:
            Audio waveform with natural pauses between sentences
        """
        if not breaks or len(breaks) <= 1:
            return wav
            
        try:
            # Calculate segment boundaries in samples
            b_ends = np.cumsum(np.array(breaks) * Synthesizer.params.hop_size)
            
            if len(b_ends) > 0 and b_ends[-1] > len(wav):
                # Scale down to fit within wav length
                scale_factor = len(wav) / b_ends[-1]
                b_ends = (b_ends * scale_factor).astype(int)
            
            b_starts = np.concatenate(([0], b_ends[:-1]))
            
            # Check for invalid segment boundaries
            valid_segments = [(start, end) for start, end in zip(b_starts, b_ends) 
                             if start < len(wav) and end <= len(wav) and start < end]
            
            if not valid_segments:
                self.logger.warning("No valid segments found for adding breaks. Returning original waveform.")
                return wav
                
            # Extract individual sentence audio segments
            wavs = [wav[start:end] for start, end in valid_segments]
            
            # Create short silences (150ms) between sentences
            silence_length = int(0.15 * Synthesizer.sample_rate)
            breaks = [np.zeros(silence_length)] * len(wavs)
            
            # Interleave audio segments with silence
            wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
            
            # Normalize to prevent clipping
            if np.abs(wav).max() > 0:
                wav_final = wav / np.abs(wav).max() * 0.97
            else:
                wav_final = wav
                
        except Exception as e:
            self.logger.error(f"Error in adding breaks: {e}. Returning original waveform.")
            wav_final = wav

        return wav_final

    def done(self):
        """Log completion of audio generation process."""
        self.logger.info("Audio Generation Successfully Completed!")
    
    @property
    def is_ready(self):
        """Check if all models are properly initialized and ready for use."""
        return all([
            self.encoder is not None,
            self.synthesizer is not None,
            self.vocoder is not None,
            self.embedder is not None
        ])