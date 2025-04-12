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
from speech_encoder_v2 import SpeechEncoderV2
from data_preprocessing import audio_preprocessing
from speech_2_text import SpeechTranslationPipeline

class Main():
    def __init__(self, original_encoder = False):
        
        warnings.filterwarnings("ignore")

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.loss_device = torch.device("cpu")
        except Exception as e:
            print(f"\nError in setting device: {e}\n Please check your CUDA installation.")
        
        self.encoder = self.__init__encoder(original_encoder)
        self.synthesizer = self.__init__synthesizer()
        self.vocoder = self.__init__vocoder()

        try:
            self.embedder = Embed(self.encoder)
        except Exception as e:
            print(f"\nError in initializing embedder: {e}\nPlease read the documentations incase of errors.")

    def __init__encoder(self, original_encoder, encoder_path: str = "models\speech_encoder_transformer_updated\encoder_073500_loss_0.0724.pt"):
        '''Initialize the encoder model'''
        try:

            if original_encoder:
                encoder = SpeechEncoder(self.device, self.loss_device)
                checkpoints = torch.load("models\speech_encoder_lstm\encoder.pt")
                encoder.load_state_dict(checkpoints['model_state'])
                return encoder

            encoder = SpeechEncoderV2(self.device, self.device)
            checkpoints = torch.load(encoder_path)
            encoder.load_state_dict(checkpoints['model_state'])
        except Exception as e:
            print(f"\nError in loading encoder: {e}\nPlease check the encoder model path.")
        return encoder
    
    def __init__synthesizer(self, synthesizer_path: Path = Path("models\synthesizer\synthesizer.pt")):
        '''Initialize the synthesizer model'''
        try:
            synthesizer = Synthesizer(synthesizer_path)
            synthesizer.load()
        except Exception as e:
            print(f"Error in loading synthesizer: {e}\nPlease check the synthesizer model path.")
        return synthesizer

    def __init__vocoder(self, vocoder_path: str = "models/vocoder/vocoder.pt"):
        '''Initialize the vocoder model'''
        try:
            vocoder = Vocoder()
            vocoder.load_model(vocoder_path)
        except Exception as e:
            print(f"\nError in loading vocoder: {e}\nPlease check the vocoder model path.")
        return vocoder

    def clone_audio(self, audio, use_vocoder: bool = False):
        print("\nModel Initializations Completed.")  
        print("\nStarting audio generation...")
        
        try:
            self.wav = preprocess_wav(audio, p.sample_rate)
        except Exception as e:
            print(f"\nError in audio preprocessing: {e}\nPlease provide a valid audio file.")

        try:
            stt_model = SpeechTranslationPipeline()
            self.text = stt_model.transcribe_audio(self.wav).split("\n")
        except Exception as e:
            print(f"\nError in speech-to-text: {e}\nPlease check the audio file or the STT model.")

        try:
            embedding, partial_embeds, _ = self.embedder.embed_utterance(self.wav, return_partials=True)
            embeddings = [embedding] * len(self.text)
        except Exception as e:
            print(f"\nError in embedding: {e}\nPlease check the audio file or the Embed model")
        
        try:
            specs = self.synthesizer.synthesize_spectrograms(self.text, embeddings)
            spec = np.concatenate(specs, axis=1)
            breaks = [spec.shape[1] for spec in specs]
        except Exception as e:
            print(f"\nError in synthesizer: {e}\nError in generating spectrograms, refer to the documentation for more details.")

        try:
            wav = self.synthesizer.griffin_lim(spec)
            wav = self.add_breaks(breaks, wav)

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
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.params.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))

        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        wav_final = wav / np.abs(wav).max() * 0.97

        return wav_final

    def done(self):
        print( "\nAudio Generation Successfully Completed!")