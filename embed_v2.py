import numpy as np
from temp import *
from typing import Union, List
import torch
import math 
from temp import audio
from data_preprocessing import audio_preprocessing_new


class EmbedV2():
    def __init__(self, encoder):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.encoder.to(self.device)

    def embed_frames_batch(self, frames_batch):
        """
        Computes embeddings for a batch of mel spectrogram.

        Args:
            frames_batch: a batch mel of spectrogram as a numpy array of float32 of shape
            (batch_size, n_frames, n_channels)

        Returns:
          the embeddings as a numpy array of float32 of shape (batch_size, model_embedding_size)
        """
        frames = torch.from_numpy(frames_batch).to(self.device)
        embed = self.encoder.forward(frames)
        return embed


    #This method is incomplete, need to fix it
    def calculate_partial_slice(self, n_samples: int, 
                           utt_frames: int,
                           min_pad: int, 
                           overlap: float) -> tuple[List[slice], List[slice]]:
        """
        Generates aligned slices for partial utterances in both waveform and mel spectrogram.
        
        Args:
            n_samples: Length of waveform in samples
            utt_frames: Number of frames per partial utterance (mel slice length)
            min_pad: Minimum required valid samples/frames (non-padded content) in any slice
            overlap: Overlap ratio between consecutive slices (0-1)
        
        Returns:
            (waveform_slices, mel_slices) tuple of slice lists
        """
        
        # Audio processing assumptions
        hop_length = 10  # 10ms frame shift at 16kHz
        win_length = 25  # 25ms window
        
        # Calculate mel parameters from audio assumptions
        n_frames = (n_samples - win_length) // hop_length + 1
        pad = (utt_frames - (n_frames % utt_frames)) % utt_frames
        n_frames += pad

        # Validate input dimensions
        utt_frames = min(utt_frames, n_frames)
        if utt_frames < 1:
            return [slice(0, n_samples)], [slice(0, n_frames)]
    
        # Calculate mel spectrogram slices first
        mel_slices = []
        mel_step = max(1, int(utt_frames * (1 - overlap)))
        
        # Generate mel slices with overlap and padding control
        start = 0
        while start + utt_frames <= n_frames:
            end = start + utt_frames
            if end > n_frames:
                # Handle last slice with padding consideration
                if (n_frames - start) < min_pad:
                    # Not enough valid frames, merge with previous
                    if mel_slices:
                        mel_slices.pop()
                    start = max(0, n_frames - utt_frames)
                end = n_frames
            mel_slices.append(slice(start, end))
            start += mel_step
        
        # Calculate corresponding waveform slices
        waveform_slices = []
        for mel_slc in mel_slices:
            # Convert frame indices to sample indices
            start_sample = max(0, mel_slc.start * hop_length)
            end_sample = min(n_samples, mel_slc.stop * hop_length + win_length)
            
            # Ensure minimum padding requirements
            valid_samples = end_sample - start_sample
            if valid_samples < min_pad * hop_length:
                # Expand slice to meet minimum requirements
                expansion = (min_pad * hop_length - valid_samples) // 2
                start_sample = max(0, start_sample - expansion)
                end_sample = min(n_samples, end_sample + expansion)
            
            waveform_slices.append(slice(start_sample, end_sample))
        
        return waveform_slices, mel_slices

    def embed_utterance(self, wav, return_partials=False):
        """"
        Computes an embedding for a single utterance
        Args:
            wav: the waveform as a numpy array of float32 of shape (n_samples,)
        Returns:
            the embedding as a numpy array of float32 of shape (model_embedding_size,)
        """
        wav_slices, mel_slices = self.calculate_partial_slice(n_samples= len(wav), utt_frames= 160, min_pad= 0.75, overlap= 0.5)
        frames = audio_preprocessing_new.wav_to_mel_spectrogram(wav = wav)
        batch_of_frames = np.array([frames[mel_slice] for mel_slice in mel_slices])

        partial_embeddings = self.embed_frames_batch(frames_batch= batch_of_frames)
        
        raw_embeddings = np.mean(partial_embeddings.detach().cpu().numpy() , axis=0)
        embeddings = raw_embeddings/np.linalg.norm(x = raw_embeddings, ord=2)
        
        if return_partials:
            return embeddings, partial_embeddings, wav_slices
        else:
            return embeddings
    
    def embed_the_speaker(self, wav: list):
        """
        Computes an embedding for a list of wavs presumambly 
        from the same speaker.
        Args:
            wav: a list of waveforms as numpy arrays of float32 of shape (n_samples,)
        Returns:
            the embeddings as a numpy array of float32 of shape (n_speakers, model_embedding_size)
        """
        raw_embeddings = np.mean([self.embed_utterance(wav= wav_) for wav_ in wav], axis = 0)
        embeddings = raw_embeddings/np.linalg.norm(x = raw_embeddings, ord=2)
        return embeddings