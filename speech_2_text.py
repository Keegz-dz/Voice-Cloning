import torch
import librosa
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional
import params as p
from data_preprocessing import audio_preprocessing

class SpeechTranslationPipeline:
    def __init__(self):
        # Initialize Whisper model for speech-to-text
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.whisper_model.config.forced_decoder_ids = None

    def transcribe_audio(self, audio) -> str:
        """
        Transcribe audio file to text using Whisper model.
        Args:
            audio (torch.Tensor or np.ndarray): Input audio tensor/array
        Returns:
            str: Transcribed text
        """
        try:
            # Convert PyTorch tensor to NumPy array if needed
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()

            # Load and preprocess audio
            input_features = self.processor(audio,sampling_rate=p.sample_rate,return_tensors="pt",language="en").input_features

            # Generate transcription
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            return transcription
        except Exception as e:
            raise Exception(f"Error in transcription: {str(e)}")


# Example usage
if __name__ == "__main__":
    
    import torchaudio
    from temp import audio
    # wav, sample_rate = torchaudio.load(r"C:\Users\anike\Downloads\harvar.flac")
    wav= audio_preprocessing.preprocess_wav(r"C:\Users\anike\Downloads\harvar.flac")
    pipeline = SpeechTranslationPipeline()
    result = pipeline.transcribe_audio(wav)
    print(result)