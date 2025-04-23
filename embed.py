import numpy as np
from temp import *
from typing import Union, List
import torch
from data_preprocessing import audio_preprocessing

# Modified work based on original code by Corentin Jemine (https://github.com/CorentinJ/Real-Time-Voice-Cloning)
# The following code is licensed under the MIT License
    
class Embed:
    """
    Audio embedding extraction class that converts audio signals into
    fixed-dimensional vector representations using a neural network encoder.
    
    This class handles the process of converting variable-length audio into consistent
    embeddings by either processing the entire audio at once or by dividing it into
    partial segments and combining their embeddings.
    
    Attributes:
        device: The computation device (GPU if available, otherwise CPU)
        encoder: The neural network model that converts spectrograms to embeddings
    """
    
    def __init__(self, encoder):
        """
        Initialize the embedding system with the specified encoder model.
        
        Args:
            encoder: A pre-trained neural network model that converts mel spectrograms to embeddings
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.encoder.to(self.device)

    def embed_frames_batch(self, frames_batch):
        """
        Compute embeddings for a batch of mel spectrograms.
        
        This is the core function that passes the spectrograms through the neural network
        to obtain fixed-dimensional vector representations.
        
        Args:
            frames_batch: Batch of mel spectrograms as a numpy array of shape
                         (batch_size, n_frames, n_channels)
                         
        Returns:
            Embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Convert numpy array to PyTorch tensor and move to appropriate device
        frames = torch.from_numpy(frames_batch).to(self.device)
        
        # Pass through the encoder network
        embed = self.encoder.forward(frames)
        return embed

    def compute_partial_slices(self, n_samples, partial_utterance_n_frames=partials_n_frames,
                              min_pad_coverage=0.75, overlap=0.5):
        """
        Determine how to divide an audio signal into overlapping segments for processing.
        
        This function computes the slice indices needed to extract partial utterances from
        both the raw waveform and its corresponding mel spectrogram. These segments can
        later be processed individually and their embeddings combined.
        
        Args:
            n_samples: Number of samples in the waveform
            partial_utterance_n_frames: Number of mel frames in each partial utterance
            min_pad_coverage: Minimum coverage threshold for the last segment (0.0-1.0)
                             If the final segment covers less than this fraction of a full segment,
                             it will be discarded unless it's the only segment
            overlap: Fraction of overlap between consecutive segments (0.0-1.0)
        
        Returns:
            A tuple containing:
            - wav_slices: List of slice objects for indexing the raw waveform
            - mel_slices: List of slice objects for indexing the mel spectrogram
        """
        # Input validation
        assert 0 <= overlap < 1, "Overlap must be between 0 and 1"
        assert 0 < min_pad_coverage <= 1, "Minimum pad coverage must be between 0 and 1"

        # Calculate conversion factors between samples and frames
        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        
        # Calculate how many frames to advance for each segment (considering overlap)
        frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

        # Compute the slices for both waveform and mel spectrogram
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
        
        for i in range(0, steps, frame_step):
            # Calculate frame range for this segment
            mel_range = np.array([i, i + partial_utterance_n_frames])
            
            # Convert frame indices to sample indices
            wav_range = mel_range * samples_per_frame
            
            # Store the slice objects
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))

        # Determine if the last segment has sufficient coverage
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        
        # Remove the last segment if coverage is insufficient (unless it's the only segment)
        if coverage < min_pad_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]

        return wav_slices, mel_slices

    def embed_utterance(self, wav, using_partials=True, return_partials=False, **kwargs):
        """
        Compute an embedding for a single utterance (audio segment).
        
        This function can process the audio in two ways:
        1. As a whole (using_partials=False)
        2. By dividing it into overlapping segments, embedding each separately,
           and then averaging (using_partials=True)
        
        The second approach is often more robust for longer utterances.
        
        Args:
            wav: Preprocessed audio waveform as a numpy array of float32
            using_partials: If True, divides the utterance into overlapping segments
                          If False, processes the entire utterance at once
            return_partials: If True, returns the embeddings of individual segments
            **kwargs: Additional arguments passed to compute_partial_slices()
        
        Returns:
            If return_partials=False:
                The utterance embedding as a normalized numpy array of shape (embedding_size,)
            
            If return_partials=True:
                A tuple containing:
                - The utterance embedding
                - The partial embeddings
                - The waveform slices corresponding to each partial
        """
        # Case 1: Process the entire utterance at once
        if not using_partials:
            # Convert waveform to mel spectrogram
            frames = audio_preprocessing.wav_to_mel_spectrogram(wav)
            
            # Add batch dimension and get embedding
            embed = self.embed_frames_batch(frames[None, ...])[0]
            
            if return_partials:
                return embed, None, None
            return embed

        # Case 2: Process with partial utterances
        
        # Step 1: Compute segmentation points
        wave_slices, mel_slices = self.compute_partial_slices(len(wav), **kwargs)
        
        # Step 2: Pad the waveform if necessary
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        # Step 3: Generate mel spectrogram for the entire utterance
        frames = audio_preprocessing.wav_to_mel_spectrogram(wav)
        
        # Step 4: Extract spectrograms for each segment
        frames_batch = np.array([frames[s] for s in mel_slices])
        
        # Step 5: Get embeddings for all segments
        partial_embeds = self.embed_frames_batch(frames_batch)

        # Step 6: Average the embeddings and normalize
        # Move tensor to CPU and convert to numpy
        raw_embed = np.mean(partial_embeds.detach().cpu().numpy(), axis=0)
        
        # L2 normalization
        embed = raw_embed / np.linalg.norm(raw_embed, 2)

        if return_partials:
            return embed, partial_embeds, wave_slices
        return embed

    def embed_speaker(self, wavs: List[np.ndarray], **kwargs):
        """
        Compute a single embedding to represent a speaker from multiple utterances.
        
        This method averages embeddings from multiple audio segments (presumably from
        the same speaker) to create a more robust speaker representation.
        
        Args:
            wavs: List of preprocessed audio waveforms as numpy arrays
            **kwargs: Additional arguments passed to embed_utterance()
            
        Returns:
            A normalized speaker embedding as a numpy array of shape (embedding_size,)
        """
        # Compute embeddings for each utterance
        utterance_embeds = [self.embed_utterance(wav, return_partials=False, **kwargs) 
                            for wav in wavs]
        
        # Average the embeddings
        raw_embed = np.mean(utterance_embeds, axis=0)
        
        # L2 normalization
        return raw_embed / np.linalg.norm(raw_embed, 2)