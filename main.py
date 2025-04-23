import torch
import librosa
import warnings
import numpy as np
from pathlib import Path
from typing import Union, List

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
        # Suppress non-critical warnings
        warnings.filterwarnings("ignore")

        # Set up computation devices (GPU if available, otherwise CPU)
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.loss_device = torch.device("cpu")
        except Exception as e:
            print(f"\nError in setting device: {e}\n Please check your CUDA installation.")
        
        # Initialize the three core neural models
        self.encoder = self.__init__encoder(original_encoder)
        self.synthesizer = self.__init__synthesizer()
        self.vocoder = self.__init__vocoder()

        # Create the embedding utility for voice identity extraction
        try:
            self.embedder = Embed(self.encoder)
        except Exception as e:
            print(f"\nError in initializing embedder: {e}\nPlease read the documentations incase of errors.")

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
                encoder = SpeechEncoder(self.device, self.loss_device)
                checkpoints = torch.load(
                    "models/speech_encoder_lstm/encoder.pt",
                    map_location=self.device
                )
                encoder.load_state_dict(checkpoints['model_state'])
                return encoder

            # Initialize improved transformer-based encoder with attention mechanism
            encoder = SpeechEncoderV2(self.device, self.device)
            checkpoints = torch.load(
                encoder_path,
                map_location=self.device
            )
            encoder.load_state_dict(checkpoints['model_state'])
            print("Successfully loaded the speaker encoder model.")
            return encoder

        except Exception as e:
            print(f"\nError in loading encoder: {e}\nPlease check the encoder model path.")
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
            synthesizer = Synthesizer(synthesizer_path)
            synthesizer.load()
        except Exception as e:
            print(f"Error in loading synthesizer: {e}\nPlease check the synthesizer model path.")
        return synthesizer

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
            vocoder = Vocoder()
            vocoder.load_model(vocoder_path)
        except Exception as e:
            print(f"\nError in loading vocoder: {e}\nPlease check the vocoder model path.")
        return vocoder

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
        print("\nModel Initializations Completed.")  
        print("\nStarting audio generation...")
        
        # Step 1: Preprocess the input audio
        try:
            self.wav = preprocess_wav(audio, p.sample_rate)
        except Exception as e:
            print(f"\nError in audio preprocessing: {e}\nPlease provide a valid audio file.")

        # Step 2: Get text to synthesize (either from parameter or via STT)
        try:
            if text is not None:
                self.text = text.split("\n")
            else:
                stt_model = SpeechTranslationPipeline()
                self.text = stt_model.transcribe_audio(self.wav).split("\n")
        except Exception as e:
            print(f"\nError in speech-to-text: {e}\nPlease check the audio file or the STT model.")

        # Step 3: Extract speaker embedding (voice identity)
        try:
            embedding, partial_embeds, _ = self.embedder.embed_utterance(self.wav, return_partials=True)
            embeddings = [embedding] * len(self.text)
        except Exception as e:
            print(f"\nError in embedding: {e}\nPlease check the audio file or the Embed model")
        
        # Step 4: Generate mel spectrograms from text with speaker embedding
        try:
            specs = self.synthesizer.synthesize_spectrograms(self.text, embeddings)
            spec = np.concatenate(specs, axis=1)
            breaks = [spec.shape[1] for spec in specs]
        except Exception as e:
            print(f"\nError in synthesizer: {e}\nError in generating spectrograms, refer to the documentation for more details.")

        # Step 5: Convert spectrograms to audio
        try:
            # First generate with Griffin-Lim algorithm (faster but lower quality)
            wav = self.synthesizer.griffin_lim(spec)
            wav = self.add_breaks(breaks, wav)

            # If requested, use neural vocoder for higher quality (slower)
            if use_vocoder:
                wav = self.vocoder.infer_waveform(spec)
                wav = self.add_breaks(breaks, wav)
                self.done()
                return wav
            
            self.done()
            return wav
        
        except Exception as e:
            print(f"\nError in decoding: {e}\n Error occurred while decoding the spectrograms to audio. Please refer to the documentation for more details.")
        

    def add_breaks(self, breaks, wav):
        """
        Add natural pauses between sentences in the generated audio.
        
        Args:
            breaks: List of spectrogram segment lengths
            wav: Audio waveform without breaks
            
        Returns:
            Audio waveform with natural pauses between sentences
        """
        # Calculate segment boundaries in samples
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.params.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))

        # Extract individual sentence audio segments
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        
        # Create short silences (150ms) between sentences
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        
        # Interleave audio segments with silence
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
        
        # Normalize to prevent clipping
        wav_final = wav / np.abs(wav).max() * 0.97

        return wav_final

    def done(self):
        """Log completion of audio generation process."""
        print("\nAudio Generation Successfully Completed!")