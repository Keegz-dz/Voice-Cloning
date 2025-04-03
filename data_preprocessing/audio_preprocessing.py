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

def preprocess_audio(waveform: torch.Tensor, orig_sr: int, debug: bool = False) -> np.ndarray:
    """
    Applies a series of audio preprocessing steps: resampling, volume normalization,
    silence trimming using VAD. Optionally includes debugging information.

    Args:
        waveform (torch.Tensor): Input audio as a PyTorch Tensor.
        orig_sr (int): Original sampling rate of the audio.
        debug (bool, optional): If True, prints debugging information and saves
                                 intermediate audio files. Defaults to False.

    Returns:
        np.ndarray: Processed audio waveform as a NumPy array.
    """
    resampled_wav = resample_audio(waveform.detach().cpu().numpy(), orig_sr, TARGET_SAMPLE_RATE)      # Convert the PyTorch tensor to a NumPy array and resample the audio
    if debug:
        debug_audio_info(resampled_wav, TARGET_SAMPLE_RATE, title="Resampled Audio")
        save_audio(resampled_wav, TARGET_SAMPLE_RATE, os.path.join(TEST_OUTPUT_DIR, "resampled.wav"))

    normalised_wav = normalize_volume(resampled_wav, AUDIO_NORM_TARGET_DBFS)
    if debug:
        debug_audio_info(normalised_wav, TARGET_SAMPLE_RATE, title="Normalised Audio")
        save_audio(normalised_wav, TARGET_SAMPLE_RATE, os.path.join(TEST_OUTPUT_DIR, "normalised.wav"))

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
    Saves an audio waveform to disk as a WAV file using the soundfile library.

    Args:
        audio (np.ndarray): Audio waveform as a NumPy array.
        sample_rate (int): Sampling rate of the audio.
        filename (str): Path to save the audio file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio, sample_rate)
    print(f"Saved audio to {filename}")

def debug_audio_info(wav: np.ndarray, sr: int, title: str = "Audio Signal"):
    """
    Prints detailed information about an audio waveform and displays relevant plots.

    Args:
        wav (np.ndarray): The audio waveform as a NumPy array.
        sr (int): The sampling rate of the audio.
        title (str, optional): A title for the plots. Defaults to "Audio Signal".
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

    # --- Plotting ---
    plt.figure(figsize=(15, 10))

    # 1. Waveform
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(wav, sr=sr)
    plt.title(f'{title} - Waveform')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 2. Power Spectrum (Magnitude Spectrum of FFT)
    n_fft = 2048  # You can adjust this
    D = np.abs(librosa.stft(wav[:n_fft], n_fft=n_fft, hop_length=n_fft // 4))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    plt.subplot(3, 2, 2)
    plt.plot(frequencies, D[:, 0]) # Plot the magnitude for the first frame
    plt.title(f'{title} - Power Spectrum (First Frame)')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xscale('log')
    plt.tight_layout()

    # 3. Spectrogram
    plt.subplot(3, 2, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav, hop_length=512)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Spectrogram')

    # 4. Mel Spectrogram (if your processing uses it)
    plt.subplot(3, 2, 4)
    n_mels = 128 # You can adjust this
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
    # Example usage: Specify an audio file path for debugging
    test_audio_file = "datasets/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
    original_waveform, original_sr = librosa.load(test_audio_file, sr=None, mono=True)
    # To enable debugging, set debug=True in the preprocess_audio call
    processed_frames = preprocess_audio(torch.tensor(original_waveform), original_sr, debug=False)
    print(f"Processed mel spectrogram shape: {processed_frames.shape}")

    # Original Audio (Unprocessed)
    debug_audio_info(original_waveform, original_sr, title="Original Audio")