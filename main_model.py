import soundfile as sf
import torch
from utils.tacotron import Tacotron
from pathlib import Path
import numpy as np
import librosa
from synthesizer import Synthesizer
from speech_encoder_v2_updated import SpeechEncoderV2
from temp.audio import preprocess_wav
from embed import Embed
from vocoder import Vocoder

def main(audio_path, intended_text):
    # Preprocess the audio
    wav, sample_rate = librosa.load(audio_path, sr=16000)
    wav = preprocess_wav(wav)

    # Initialize and load speaker encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    encoder = SpeechEncoderV2(device, device)
    checkpoints = torch.load("models/speech_encoder_transformer_updated/encoder_014000_loss_0.4987.pt")
    encoder.load_state_dict(checkpoints['model_state'])
    embedder = Embed(encoder)

    # Generate speaker embeddings
    embedding, partial_embeds, _ = embedder.embed_utterance(wav, return_partials=True)
    # If multiple lines are provided, split into a list. Otherwise, wrap the text in a list.
    text = intended_text.split("\n") 
    embeddings = [embedding] * len(text)

    # Synthesize spectrograms
    synthesizer_model_path = Path("models/synthesizer/synthesizer.pt")
    synthesizer = Synthesizer(synthesizer_model_path)
    synthesizer.load()
    specs = synthesizer.synthesize_spectrograms(text, embeddings)
    spec = np.concatenate(specs, axis=1)
    breaks = [spec.shape[1] for spec in specs]

    # Generate WAV using Vocoder
    vocoder = Vocoder()
    vocoder.load_model("models/vocoder/vocoder.pt")
    wav_new = vocoder.infer_waveform(spec)

    # Split and recombine the waveform based on spectrogram breaks
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.params.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav_new[start:end] for start, end in zip(b_starts, b_ends)]
    silence_breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    wav_vocoder = np.concatenate([segment for w, b in zip(wavs, silence_breaks) for segment in (w, b)])
    wav_vocoder = wav_vocoder / np.abs(wav_vocoder).max() * 0.97

    # Save the final waveform to a file
    sf.write('vocoder_output.wav', wav_vocoder, Synthesizer.sample_rate)

if __name__ == "__main__":
    # Set the path to your .flac audio file and the intended text here
    audio_path = "2764-36616-0000.flac"  # Replace with the path to your .flac file
    intended_text = ("I went to the zoo with my family, we saw birds, lions and various other animals. "
                     "WE enjoyed the day.")
    main(audio_path, intended_text)
