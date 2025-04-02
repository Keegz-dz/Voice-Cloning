import numpy as np
import librosa
import torch
import webrtcvad
import struct
from scipy.ndimage import binary_dilation
import os
import soundfile as sf

# Constants 
TARGET_SAMPLE_RATE = 16000          
AUDIO_NORM_TARGET_DBFS = -30        # Desired loudness level in dBFS
VAD_WINDOW_LENGTH = 30              # VAD window length in milliseconds 
VAD_MOVING_AVERAGE_WIDTH = 7        # Width for smoothing VAD results
VAD_MAX_SILENCE_LENGTH = 6          # Maximum consecutive silent windows allowed
INT16_MAX = (2 ** 15) - 1           # Maximum 16-bit integer value
TEST_OUTPUT_DIR = "test"            # Directory for debug audio output

def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resamples the waveform to the target sampling rate."""
    if waveform.ndim > 1:
        waveform = waveform[0]
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return waveform

def normalize_volume(wav: np.ndarray, target_dBFS: float = AUDIO_NORM_TARGET_DBFS) -> np.ndarray:
    """Normalizes the audio waveform to a target dBFS level."""
    rms = np.sqrt(np.mean((wav * INT16_MAX) ** 2))
    wave_dBFS = 20 * np.log10(rms / INT16_MAX + 1e-6)
    dBFS_change = target_dBFS - wave_dBFS
    factor = 10 ** (dBFS_change / 20)
    return wav * factor

def trim_long_silences(wav: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Trims long periods of silence using voice activity detection (VAD).
    Returns only the voiced parts of the waveform.
    """
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
    # Unless required by applicable law or agreed to in writing, software distributed under the License 
    # is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express 
    # or implied. See the License for the specific language governing permissions and limitations 
    # under the License.

    samples_per_window = int((VAD_WINDOW_LENGTH * sample_rate) / 1000)
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * INT16_MAX)).astype(np.int16))
    vad = webrtcvad.Vad(mode=3)
    voice_flags = []
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        is_speech = vad.is_speech(pcm_wave[window_start*2: window_end*2], sample_rate=sample_rate)
        voice_flags.append(1 if is_speech else 0)
    voice_flags = np.array(voice_flags, dtype=np.float32)
    
    def moving_average(arr, width):
        return np.convolve(arr, np.ones(width) / width, mode='same')
    
    smoothed_flags = moving_average(voice_flags, VAD_MOVING_AVERAGE_WIDTH)
    audio_mask = np.round(smoothed_flags).astype(bool)
    audio_mask = binary_dilation(audio_mask, np.ones(VAD_MAX_SILENCE_LENGTH + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    if len(audio_mask) != len(wav):
        if len(audio_mask) < len(wav):
            pad_length = len(wav) - len(audio_mask)
            audio_mask = np.concatenate([audio_mask, np.zeros(pad_length, dtype=bool)])
        else:
            audio_mask = audio_mask[:len(wav)]
    return wav[audio_mask]

def preprocess_audio(waveform: torch.Tensor, orig_sr: int) -> np.ndarray:
    """
    Applies resampling, volume normalization, and silence trimming to the audio.
    
    :param waveform: Input audio as a torch.Tensor.
    :param orig_sr: Original sampling rate of the audio.
    :return: Preprocessed audio as a numpy array.
    """
    wav = resample_audio(waveform.detach().cpu().numpy(), orig_sr, TARGET_SAMPLE_RATE)
    wav = normalize_volume(wav, AUDIO_NORM_TARGET_DBFS)
    wav = trim_long_silences(wav, TARGET_SAMPLE_RATE)
    return wav

def save_audio(audio: np.ndarray, sample_rate: int, filename: str):
    """
    Saves an audio waveform to disk using soundfile.
    
    :param audio: Audio waveform as a numpy array.
    :param sample_rate: Sampling rate of the audio.
    :param filename: Path to save the audio file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio, sample_rate)
    print(f"Saved audio to {filename}")

def debug_loader(audio_file: str):
    """
    Debug function to test the preprocessing pipeline on a single audio file.
    
    :param audio_file: Path to the audio file to test.
    """
    print(f"Loading audio file: {audio_file}")
    waveform, orig_sr = librosa.load(audio_file, sr=None, mono=True)
    print(f"Original waveform shape: {waveform.shape}, Sample rate: {orig_sr}")
    try:
        preprocessed_wav = preprocess_audio(torch.tensor(waveform), orig_sr)
        print(f"Preprocessed waveform shape: {preprocessed_wav.shape}")
        original_filename = os.path.join(TEST_OUTPUT_DIR, "original.wav")
        save_audio(waveform, orig_sr, original_filename)
        processed_filename = os.path.join(TEST_OUTPUT_DIR, "preprocessed.wav")
        save_audio(preprocessed_wav, TARGET_SAMPLE_RATE, processed_filename)
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    test_audio_file = "data/train-clean-100/19/198/19-198-0000.flac"
    debug_loader(test_audio_file)
