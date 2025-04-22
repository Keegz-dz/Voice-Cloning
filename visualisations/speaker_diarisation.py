""" This script demonstrates how to perform speaker diarization using the SpeechEncoder toolkit.
It extracts reference segments for known speakers from an audio interview, computes continuous
embeddings for the entire interview, and then compares the continuous embeddings with the
reference speaker embeddings using a cosine similarity.

The final output is an interactive animation that displays the similarity curves over time,
while playing the audio in sync with the plot.
"""
import sys
sys.path.append('.')

import torch
import librosa
import torchaudio
import numpy as np
import matplotlib
from pathlib import Path
from typing import Union, List

# Import necessary modules
from speech_encoder_v2 import SpeechEncoderV2
from params import *
from utils import *
from visualisations import *
from embed import Embed
from data_preprocessing import audio_preprocessing
from speaker_diarisation_utils import interactive_diarization

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # Load audio file with error handling
    try:
        # Try different path formats to handle both OS types
        try:
            waveform, sample_rate = torchaudio.load("visualisations/demo_speaker_diarisation.mp3")
        except:
            waveform, sample_rate = torchaudio.load("visualisations\\demo_speaker_diarisation.mp3")
            
        print(f"Audio loaded successfully. Sample rate: {sample_rate}Hz, Duration: {waveform.shape[1]/sample_rate:.2f}s")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("Please ensure the audio file path is correct.")
        return
    
    # Preprocess audio - convert to mono and resample if needed
    if waveform.shape[0] > 1:
        print("Converting stereo to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sample_rate != sampling_rate:
        print(f"Resampling from {sample_rate}Hz to {sampling_rate}Hz...")
        waveform = torchaudio.transforms.Resample(sample_rate, sampling_rate)(waveform)
    
    # Apply audio preprocessing
    print("Applying audio preprocessing...")
    wav = audio_preprocessing.preprocess_audio(waveform, sampling_rate)
    
    # Define reference speaker segments
    segments = [[0, 5.5], [6.5, 12], [17, 25]]
    speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
    
    # Extract speaker reference segments
    print("Extracting reference segments for speakers...")
    speaker_wavs = []
    for i, (start, end) in enumerate(segments):
        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)
        if start_sample >= wav.shape[0] or end_sample > wav.shape[0]:
            print(f"Warning: Segment {i} ({start}-{end}s) exceeds audio length. Adjusting...")
            end_sample = min(end_sample, wav.shape[0])
            start_sample = min(start_sample, end_sample-1)
        
        speaker_wavs.append(wav[start_sample:end_sample])
    
    # Load model
    print("Loading speech encoder model...")
    encoder = SpeechEncoderV2(device, torch.device("cpu"))
    
    try:
        checkpoints = torch.load(
            "models/speech_encoder_transformer/encoder(0.096).pt",
            map_location=device
        )
        encoder.load_state_dict(checkpoints['model_state'])
        encoder.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create embedder
    embedder = Embed(encoder)
    
    # Compute embeddings with progress indication
    print("Computing continuous embeddings for the full audio...")
    encoder.use_nested_tensor = False
    
    # Process in batches if audio is long
    if len(wav) > 10 * sampling_rate * 60:  # If longer than 10 minutes
        print("Audio is long, processing in chunks...")
        chunk_size = 5 * 60 * sampling_rate  # 5-minute chunks
        embeddings_list = []
        splits_list = []
        
        for i in range(0, len(wav), chunk_size):
            chunk_end = min(i + chunk_size, len(wav))
            print(f"Processing chunk {i/sampling_rate/60:.1f}m - {chunk_end/sampling_rate/60:.1f}m...")
            wav_chunk = wav[i:chunk_end]
            _, chunk_embeds, chunk_splits = embedder.embed_utterance(wav_chunk, return_partials=True)
            
            # Adjust splits to account for offset
            adjusted_splits = [slice(s.start + i, s.stop + i) for s in chunk_splits]
            
            embeddings_list.append(chunk_embeds)
            splits_list.extend(adjusted_splits)
        
        cont_embeds = torch.cat(embeddings_list, dim=0)
        wav_splits = splits_list
    else:
        _, cont_embeds, wav_splits = embedder.embed_utterance(wav, return_partials=True)
    
    print(f"Created {len(wav_splits)} embedding segments")
    
    # Compute reference speaker embeddings
    print("Computing reference speaker embeddings...")
    speaker_embeds = []
    for i, speaker_wav in enumerate(speaker_wavs):
        print(f"Processing speaker: {speaker_names[i]}")
        speaker_embed = torch.tensor(embedder.embed_utterance(speaker_wav), device=device)
        speaker_embeds.append(speaker_embed)
    
    # Compute similarity
    print("Computing similarities...")
    similarity_dict = {}
    for name, speaker_embed in zip(speaker_names, speaker_embeds):
        # Compute similarity and convert to numpy array
        similarity = (cont_embeds @ speaker_embed).detach().cpu().numpy()
        similarity_dict[name] = similarity
    
    # Run interactive visualization
    print("Starting interactive visualization...")
    interactive_diarization(similarity_dict, wav, wav_splits, x_crop=7, show_time=True)


if __name__ == "__main__":
    main()