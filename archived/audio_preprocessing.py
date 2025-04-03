import numpy as np
import librosa
import torch
import webrtcvad
import struct
from scipy.ndimage import binary_dilation
import os
import soundfile as sf

# =============================================================================
# Constants
# =============================================================================

# Audio processing constants
TARGET_SAMPLE_RATE = 16000
AUDIO_NORM_TARGET_DBFS = -30
VAD_WINDOW_LENGTH = 30  # milliseconds
VAD_MOVING_AVERAGE_WIDTH = 7
VAD_MAX_SILENCE_LENGTH = 6
INT16_MAX = (2 ** 15) - 1

# Mel spectrogram extraction constants
MEL_WINDOW_LENGTH = 25  # milliseconds
MEL_WINDOW_STEP = 10    # milliseconds
MEL_N_CHANNELS = 40

# Debugging constant
TEST_OUTPUT_DIR = "test"

# =============================================================================
# Audio Processing Functions
# =============================================================================

def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Resamples the input audio waveform to the specified target sampling rate.

    Args:
        waveform (np.ndarray): The input audio waveform as a NumPy array.
        orig_sr (int): The original sampling rate of the input waveform.
        target_sr (int, optional): The desired target sampling rate. Defaults to TARGET_SAMPLE_RATE.

    Returns:
        np.ndarray: The resampled audio waveform.
    """
    if waveform.ndim > 1:           # Handle multi-channel audio by taking only the first channel
        waveform = waveform[0]
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        
    return waveform


def normalize_volume(wav: np.ndarray, target_dBFS: float = AUDIO_NORM_TARGET_DBFS) -> np.ndarray:
    """
    Normalizes the volume of the audio waveform to the specified target dBFS level.

    Args:
        wav (np.ndarray): The input audio waveform as a NumPy array.
        target_dBFS (float, optional): The desired target loudness level in dBFS. Defaults to AUDIO_NORM_TARGET_DBFS.

    Returns:
        np.ndarray: The volume-normalized audio waveform.
    """
    rms = np.sqrt(np.mean((wav * INT16_MAX) ** 2))
    wave_dBFS = 20 * np.log10(rms / INT16_MAX + 1e-6)
    dBFS_change = target_dBFS - wave_dBFS
    factor = 10 ** (dBFS_change / 20)           # Calculate the scaling factor to apply to the waveform
    
    return wav * factor


def trim_long_silences(wav: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Trims long periods of silence from the audio waveform using voice activity detection (VAD).
    Only the voiced parts of the waveform are returned.

    Args:
        wav (np.ndarray): The input audio waveform as a NumPy array.
        sample_rate (int, optional): The sampling rate of the audio. Defaults to TARGET_SAMPLE_RATE.

    Returns:
        np.ndarray: The audio waveform with long silences trimmed.
    """
    samples_per_window = int((VAD_WINDOW_LENGTH * sample_rate) / 1000)
    # Ensure the waveform length is a multiple of the window size for VAD
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    # Convert to 16-bit PCM for WebRTC VAD
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * INT16_MAX)).astype(np.int16))
    vad = webrtcvad.Vad(mode=3)
    voice_flags = []
    # Iterate through the waveform in VAD window steps
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        # Check if the current window contains speech
        is_speech = vad.is_speech(pcm_wave[window_start * 2: window_end * 2], sample_rate=sample_rate)
        voice_flags.append(1 if is_speech else 0)
    voice_flags = np.array(voice_flags, dtype=np.float32)

    # Apply a moving average filter to smooth the VAD decisions
    def moving_average(arr, width):
        return np.convolve(arr, np.ones(width) / width, mode='same')

    smoothed_flags = moving_average(voice_flags, VAD_MOVING_AVERAGE_WIDTH)
    # Create a binary mask for speech segments
    audio_mask = np.round(smoothed_flags).astype(bool)
    # Dilate the mask to include short silences between speech
    audio_mask = binary_dilation(audio_mask, np.ones(VAD_MAX_SILENCE_LENGTH + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    # Ensure the mask length matches the waveform length
    if len(audio_mask) != len(wav):
        if len(audio_mask) < len(wav):
            pad_length = len(wav) - len(audio_mask)
            audio_mask = np.concatenate([audio_mask, np.zeros(pad_length, dtype=bool)])
        else:
            audio_mask = audio_mask[:len(wav)]
            
    return wav[audio_mask]


def wav_to_mel_spectrogram(wav: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Computes a mel spectrogram (not in log-scale) from the input audio waveform.
    This function mirrors the mel spectrogram computation in the original audio module.

    Args:
        wav (np.ndarray): The input audio waveform as a NumPy array.
        sample_rate (int, optional): The sampling rate of the audio. Defaults to TARGET_SAMPLE_RATE.

    Returns:
        np.ndarray: The mel spectrogram as a NumPy array, with time as the first dimension.
    """
    n_fft = int(sample_rate * MEL_WINDOW_LENGTH / 1000)
    hop_length = int(sample_rate * MEL_WINDOW_STEP / 1000)
    # Compute the mel spectrogram using librosa
    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=MEL_N_CHANNELS
    )
    
    # Transpose so that time is the first dimension (frames x frequency bins)
    return mel_spec.astype(np.float32).T


# =============================================================================
# Main Preprocessing Function
# =============================================================================

def preprocess_audio(waveform: torch.Tensor, orig_sr: int) -> np.ndarray:
    """
    Applies a series of audio preprocessing steps: resampling, volume normalization,
    silence trimming using VAD, and conversion to a mel spectrogram.

    Args:
        waveform (torch.Tensor): Input audio as a PyTorch Tensor.
        orig_sr (int): Original sampling rate of the audio.

    Returns:
        np.ndarray: Processed mel spectrogram as a NumPy array (frames x frequency bins).
    """
    wav = resample_audio(waveform.detach().cpu().numpy(), orig_sr, TARGET_SAMPLE_RATE)      # Convert the PyTorch tensor to a NumPy array and resample the audio
    wav = normalize_volume(wav, AUDIO_NORM_TARGET_DBFS)
    wav = trim_long_silences(wav, TARGET_SAMPLE_RATE)
    # Convert the processed waveform to mel spectrogram frames
    frames = wav_to_mel_spectrogram(wav, TARGET_SAMPLE_RATE)
    
    return frames

# =============================================================================
# Utility and Debugging Functions
# =============================================================================

def save_audio(audio: np.ndarray, sample_rate: int, filename: str):
    """
    Saves an audio waveform to disk as a WAV file using the soundfile library.

    Args:
        audio (np.ndarray): Audio waveform as a NumPy array.
        sample_rate (int): Sampling rate of the audio.
        filename (str): Path to save the audio file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio, sample_rate)
    print(f"Saved audio to {filename}")

def debug_loader(audio_file: str):
    """
    Debug function to load an audio file, preprocess it, and save intermediate results.

    Args:
        audio_file (str): Path to the audio file to test.
    """
    print(f"Loading audio file: {audio_file}")
    waveform, orig_sr = librosa.load(audio_file, sr=None, mono=True)
    print(f"Original waveform shape: {waveform.shape}, Sample rate: {orig_sr}")
    try:
        processed_frames = preprocess_audio(torch.tensor(waveform), orig_sr)
        print(f"Processed mel spectrogram shape: {processed_frames.shape}")
        original_filename = os.path.join(TEST_OUTPUT_DIR, "original.wav")
        save_audio(waveform, orig_sr, original_filename)
        processed_filename = os.path.join(TEST_OUTPUT_DIR, "processed.wav")
        save_audio(trim_long_silences(normalize_volume(resample_audio(waveform, orig_sr)), TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE, processed_filename)
        # Save the processed mel spectrogram frames for inspection
        np.save(os.path.join(TEST_OUTPUT_DIR, "processed_frames.npy"), processed_frames)
        print("Saved processed frames as numpy array.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == "__main__":
    # Example usage: Specify an audio file path for debugging
    test_audio_file = "datasets/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
    debug_loader(test_audio_file)