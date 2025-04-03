""" This script demonstrates how to perform speaker diarization using the SpeechEncoder toolkit.
It extracts reference segments for known speakers from an audio interview, computes continuous
embeddings for the entire interview, and then compares the continuous embeddings with the
reference speaker embeddings using a dot product (cosine similarity).

The final output is an interactive animation that displays the similarity curves over time,
while playing the audio in sync with the plot.
"""
import sys
sys.path.append('.')

import sounddevice
from speech_encoder_v2 import SpeechEncoderV2
from params import *
from utils import *
from pathlib import Path
import torch
import librosa
import torchaudio
import numpy as np
# from temp import *
from typing import Union, List
from visualisations import *
from embed import Embed
from data_preprocessing import audio_preprocessing_new

# waveform, sample_rate = torchaudio.load("visualisations/demo_speaker_diarisation.mp3")      # MacOS
waveform, sample_rate = torchaudio.load("visualisations\demo_speaker_diarisation.mp3")    # Windows
wav = audio_preprocessing_new.preprocess_audio(waveform, sample_rate)

# Cut reference segments from the interview. Each segment (given in seconds) is assumed to contain a single speaker.
segments = [[0, 5.5], [6.5, 12], [17, 25]]
speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]     # Convert time intervals to sample indices and slice the waveform accordingly.

# ---------------------------
# Compute Embeddings and Similarity
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_device = torch.device("cpu")

encoder = SpeechEncoderV2(device,device)
checkpoints = torch.load("models\speech_encoder_transformer\encoder(0.096).pt") 
encoder.load_state_dict(checkpoints['model_state'])
encoder.eval()

print("Running the continuous embedding on CPU, this might take a while...")

embedder = Embed(encoder)

# Compute a continuous embedding for the full interview.
_, cont_embeds, wav_splits = embedder.embed_utterance(wav, return_partials=True)

# Compute a single embedding for each reference speaker.
speaker_embeds = [torch.tensor(embedder.embed_utterance(speaker_wav), device=device) for speaker_wav in speaker_wavs]

# Ensure continuous embeddings are on the same device.
cont_embeds = cont_embeds.to(device)

# Compute similarity using proper device management.
similarity_dict = {
    name: (cont_embeds @ speaker_embed).detach().cpu().numpy()
    for name, speaker_embed in zip(speaker_names, speaker_embeds)
}

# ---------------------------
# Run Interactive Diarization Demo
# ---------------------------
interactive_diarization(similarity_dict, wav, wav_splits)

