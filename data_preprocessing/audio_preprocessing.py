import numpy as np
import librosa
import torch
import webrtcvad
import struct
from scipy.ndimage import binary_dilation
import os
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display

# =============================================================================
# Constants
# =============================================================================

# Audio processing parameters
TARGET_SAMPLE_RATE = 16000  # 16kHz is standard for speech recognition tasks
AUDIO_NORM_TARGET_DBFS = -30  # Standard level for speech processing and neural networks
INT16_MAX = (2 ** 15) - 1  # Maximum value for 16-bit audio

# Voice Activity Detection (VAD) parameters
VAD_WINDOW_LENGTH = 30  # Window size in ms for voice detection
VAD_MOVING_AVERAGE_WIDTH = 7  # Smoothing factor for VAD decisions
VAD_MAX_SILENCE_LENGTH = 6  # Max number of consecutive silence frames to keep

# Mel spectrogram extraction parameters
MEL_WINDOW_LENGTH = 25  # STFT window length in ms (standard for speech)
MEL_WINDOW_STEP = 10    # STFT hop length in ms (60% overlap)
MEL_N_CHANNELS = 40     # Number of mel bands (dimensionality of features)

# Debugging directory
TEST_OUTPUT_DIR = "test"

# =============================================================================
# Audio Processing Functions
# =============================================================================

def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Resamples audio to a consistent sampling rate for subsequent processing.
    
    Args:
        waveform (np.ndarray): The input audio waveform as a NumPy array
        orig_sr (int): The original sampling rate of the input waveform
        target_sr (int): The desired target sampling rate
    
    Returns:
        np.ndarray: The resampled audio waveform
    """
    if waveform.ndim > 1:  # Convert multi-channel to mono by using first channel
        waveform = waveform[0]
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    return waveform


def normalize_volume(wav: np.ndarray, target_dBFS: float = AUDIO_NORM_TARGET_DBFS) -> np.ndarray:
    """
    Normalizes audio volume to a consistent level, crucial for stable feature extraction.
    
    Higher target_dBFS values (closer to 0) result in louder audio, while lower values
    result in quieter audio. -30 dBFS is a standard level for speech processing.
    
    Args:
        wav (np.ndarray): The input audio waveform, normalized to [-1, 1]
        target_dBFS (float): The target loudness level in dBFS
    
    Returns:
        np.ndarray: Volume-normalized audio waveform
    """
    rms = np.sqrt(np.mean((wav * INT16_MAX) ** 2))
    wave_dBFS = 20 * np.log10(rms / INT16_MAX + 1e-6)
    dBFS_change = target_dBFS - wave_dBFS
    factor = 10 ** (dBFS_change / 20)  # Calculate scaling factor to apply

    return wav * factor


def trim_long_silences(wav: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Removes silent segments from audio using WebRTC Voice Activity Detection.
    
    The process:
    1. Divides audio into short frames (30ms)
    2. Uses WebRTC VAD (mode 3 = most aggressive) to classify frames as speech/non-speech
    3. Applies smoothing to prevent choppy results
    4. Uses binary dilation to preserve short silences between words
    
    Args:
        wav (np.ndarray): Input waveform normalized to [-1, 1]
        sample_rate (int): Audio sampling rate, must be 8kHz, 16kHz, 32kHz or 48kHz
                          (WebRTC VAD requirement)
    
    Returns:
        np.ndarray: Audio with long silences removed, preserving short pauses
    """
    # Modified work based on original code using Resemblyzer (https://github.com/resemble-ai/Resemblyzer)
    # The following code is licensed under the MIT License
    
    # Calculate samples per window based on VAD_WINDOW_LENGTH constant
    samples_per_window = int((VAD_WINDOW_LENGTH * sample_rate) / 1000)
    
    # Ensure the waveform length is a multiple of the window size for VAD
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert float waveform to 16-bit PCM format required by WebRTC VAD
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * INT16_MAX)).astype(np.int16))
    
    # Initialize WebRTC VAD with highest aggressiveness (mode 3)
    vad = webrtcvad.Vad(mode=3)
    voice_flags = []
    
    # Process each window and determine if it contains speech
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        is_speech = vad.is_speech(pcm_wave[window_start * 2: window_end * 2], sample_rate=sample_rate)
        voice_flags.append(1 if is_speech else 0)
    
    voice_flags = np.array(voice_flags, dtype=np.float32)

    # Apply a moving average filter to smooth the VAD decisions
    def moving_average(arr, width):
        return np.convolve(arr, np.ones(width) / width, mode='same')

    # Smooth the voice flags and create a binary mask
    smoothed_flags = moving_average(voice_flags, VAD_MOVING_AVERAGE_WIDTH)
    audio_mask = np.round(smoothed_flags).astype(bool)
    
    # Dilate the mask to include short silences between speech
    # This prevents choppy results by keeping short pauses intact
    audio_mask = binary_dilation(audio_mask, np.ones(VAD_MAX_SILENCE_LENGTH + 1))
    
    # Expand mask to match the original waveform length
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    # Ensure the mask length matches the waveform length
    if len(audio_mask) != len(wav):
        if len(audio_mask) < len(wav):
            pad_length = len(wav) - len(audio_mask)
            audio_mask = np.concatenate([audio_mask, np.zeros(pad_length, dtype=bool)])
        else:
            audio_mask = audio_mask[:len(wav)]

    # Apply the mask to keep only speech segments
    return wav[audio_mask]


def wav_to_mel_spectrogram(wav: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Converts waveform to mel spectrogram features suitable for neural networks.
    
    Mel spectrograms better represent how humans perceive sound by:
    1. Using mel scale (logarithmic) instead of linear frequency scale
    2. Focusing more resolution on lower frequencies (where speech information is dense)
    
    Args:
        wav (np.ndarray): The input audio waveform
        sample_rate (int): The sampling rate of the audio
    
    Returns:
        np.ndarray: The mel spectrogram as a NumPy array (frames × frequency bins)
    """
    # Calculate FFT and hop parameters in samples
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

    # Transpose to get time as the first dimension (frames × frequency bins)
    return mel_spec.astype(np.float32).T


# =============================================================================
# Main Preprocessing Function
# =============================================================================

def preprocess_audio(waveform: torch.Tensor, orig_sr: int, debug: bool = False) -> np.ndarray:
    """
    Complete audio preprocessing pipeline for speech processing.
    
    Processing steps:
    1. Resampling to 16kHz (standard for speech processing)
    2. Volume normalization to -30 dBFS (ensures consistent input levels)
    3. Silence removal using VAD (reduces computation and improves feature quality)
    
    These steps are crucial for creating consistent training data that focuses
    on the speech content rather than recording conditions.
    
    Args:
        waveform (torch.Tensor): Raw input audio as PyTorch tensor
        orig_sr (int): Original sampling rate
        debug (bool): Whether to save intermediate files and plot diagnostics
    
    Returns:
        np.ndarray: Processed audio waveform, ready for feature extraction
    """
    # Convert PyTorch tensor to NumPy array and resample
    resampled_wav = resample_audio(waveform.detach().cpu().numpy(), orig_sr, TARGET_SAMPLE_RATE)
    if debug:
        debug_audio_info(resampled_wav, TARGET_SAMPLE_RATE, title="Resampled Audio")
        save_audio(resampled_wav, TARGET_SAMPLE_RATE, os.path.join(TEST_OUTPUT_DIR, "resampled.wav"))

    # Normalize volume to consistent level
    normalised_wav = normalize_volume(resampled_wav, AUDIO_NORM_TARGET_DBFS)
    if debug:
        debug_audio_info(normalised_wav, TARGET_SAMPLE_RATE, title="Normalised Audio")
        save_audio(normalised_wav, TARGET_SAMPLE_RATE, os.path.join(TEST_OUTPUT_DIR, "normalised.wav"))

    # Remove silence using Voice Activity Detection
    trimmed_wav = trim_long_silences(normalised_wav, TARGET_SAMPLE_RATE)
    if debug:
        debug_audio_info(trimmed_wav, TARGET_SAMPLE_RATE, title="Trimmed Audio")
        save_audio(trimmed_wav, TARGET_SAMPLE_RATE, os.path.join(TEST_OUTPUT_DIR, "trimmed.wav"))

    return trimmed_wav

# =============================================================================
# Utility and Debugging Functions
# =============================================================================

def save_audio(audio: np.ndarray, sample_rate: int, filename: str):
    """
    Saves an audio waveform to disk as a WAV file.
    
    Args:
        audio (np.ndarray): Audio waveform as a NumPy array
        sample_rate (int): Sampling rate of the audio
        filename (str): Path to save the audio file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio, sample_rate)
    print(f"Saved audio to {filename}")


def debug_audio_info(wav: np.ndarray, sr: int, title: str = "Audio Signal"):
    """
    Displays detailed diagnostic information and visualizations for audio analysis.
    
    Creates visualizations of:
    1. Waveform (amplitude vs time)
    2. Power spectrum (frequency analysis)
    3. Spectrogram (frequency content over time)
    4. Mel spectrogram (perceptually-weighted frequency content)
    
    Args:
        wav (np.ndarray): The audio waveform as a NumPy array
        sr (int): The sampling rate of the audio
        title (str): A title for the plots
    """
    print(f"\n--- Debugging Information for: {title} ---")
    print(f"Shape of the waveform: {wav.shape}")
    print(f"Data type of the waveform: {wav.dtype}")
    print(f"Sampling rate: {sr} Hz")
    print(f"Minimum value: {np.min(wav)}")
    print(f"Maximum value: {np.max(wav)}")
    print(f"Mean value: {np.mean(wav)}")
    print(f"Standard deviation: {np.std(wav)}")
    print(f"Number of samples: {len(wav)}")
    print(f"Duration: {len(wav) / sr:.2f} seconds")

    plt.figure(figsize=(15, 10))

    # 1. Waveform visualization
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(wav, sr=sr)
    plt.title(f'{title} - Waveform')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 2. Power Spectrum visualization
    n_fft = 2048  
    D = np.abs(librosa.stft(wav[:n_fft], n_fft=n_fft, hop_length=n_fft // 4))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    plt.subplot(3, 2, 2)
    plt.plot(frequencies, D[:, 0])  # Plot the magnitude for the first frame
    plt.title(f'{title} - Power Spectrum (First Frame)')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale('log')
    plt.tight_layout()

    # 3. Spectrogram visualization
    plt.subplot(3, 2, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Spectrogram')

    # 4. Mel Spectrogram visualization
    plt.subplot(3, 2, 4)
    n_mels = 40
    mel_spectrogram = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Mel Spectrogram')
    plt.tight_layout()

    plt.show()

# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == "__main__":
    # Test the preprocessing pipeline on a sample file
    test_audio_file = "datasets/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
    original_waveform, original_sr = librosa.load(test_audio_file, sr=None, mono=True)
    
    # Display original audio information
    debug_audio_info(original_waveform, original_sr, title="Original Audio")
    
    # Process the audio and visualize each step
    processed_wav = preprocess_audio(torch.tensor(original_waveform), original_sr, debug=True)
    
    # Convert to mel spectrogram features
    processed_frames = wav_to_mel_spectrogram(processed_wav, TARGET_SAMPLE_RATE)
    print(f"Processed mel spectrogram shape: {processed_frames.shape}")