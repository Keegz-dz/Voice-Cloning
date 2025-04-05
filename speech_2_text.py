import torch
import librosa
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional
import params as p

class SpeechTranslationPipeline:
    def __init__(self, target_language: str = "hi", translate = False):
        """
        Initialize the speech translation pipeline.
        Args:
            target_language (str): Target language code (e.g., 'hi' for Hindi, 'es' for Spanish)
        """
        # Initialize Whisper model for speech-to-text
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.whisper_model.config.forced_decoder_ids = None

        if translate:
            self.target_language = target_language
            self.translation_model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
            self.translator = AutoModelForSeq2SeqLM.from_pretrained(self.translation_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.translation_model_name, use_fast=False)

    def transcribe_audio(self, audio) -> str:
        """
        Transcribe audio file to text using Whisper model.
        Args:
            audio_path (str): Path to the audio file
        Returns:
            str: Transcribed text
        """
        try:
            # Load and preprocess audio
            input_features = self.processor(
                audio, 
                sampling_rate=p.sample_rate, 
                return_tensors="pt",
                language="en"
            ).input_features

            # Generate transcription
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription
        except Exception as e:
            raise Exception(f"Error in transcription: {str(e)}")

    def translate_text(self, text: str) -> str:
        """
        Translate text to target language.
        Args:
            text (str): Text to translate
        Returns:
            str: Translated text
        """
        try:
            inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
            translated_ids = self.translator.generate(**inputs)
            translated_text = self.tokenizer.batch_decode(
                translated_ids, 
                skip_special_tokens=True
            )[0]
            
            return translated_text
        except Exception as e:
            raise Exception(f"Error in translation: {str(e)}")

    def process_audio(self, 
                     input_audio_path: str, translate = False) -> dict:
        """
        Process audio through the complete pipeline to generate Text from Speech and Translate it if needed.
        Args:
            input_audio_path (str): Path to input audio file
            output_audio_path (str, optional): Path to save translated audio
        Returns:
            dict: Dictionary containing original transcript and translation
        """
        try:
            # Generate default output path if not provided
            if output_audio_path is None:
                base_path = os.path.splitext(input_audio_path)[0]
                output_audio_path = f"{base_path}_translated_{self.target_language}.mp3"

            # Execute pipeline
            transcript = self.transcribe_audio(input_audio_path)
            if translate:
                translation = self.translate_text(transcript)
                return {
                    "original_transcript": transcript,
                    "translation": translation,
                    }

            return transcript
        
        except Exception as e:
            raise Exception(f"Error in pipeline execution: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline with Hindi as target language
    pipeline = SpeechTranslationPipeline(target_language="hi")
    
    # Process an audio file
    try:
        result = pipeline.process_audio(
            input_audio_path="librispeech_809_gt.wav",
            output_audio_path="translated_output.mp3"
        )
        
        print("Original Transcript:", result["original_transcript"])
        print("Translated Text:", result["translation"])
    except Exception as e:
        print(f"Error: {str(e)}") 