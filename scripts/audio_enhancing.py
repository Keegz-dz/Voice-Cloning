import torch
import torchaudio
import numpy as np
import soundfile as sf
import librosa
import io
import os
import pyaudio
from pedalboard import Pedalboard, NoiseGate, Compressor, LowShelfFilter, Gain
import noisereduce as nr
import sounddevice as sd
from typing import Union, Tuple
import torch
import numpy as np

def torch_to_librosa(tensor: torch.Tensor, sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Convert a PyTorch audio tensor to librosa-style format (numpy array, sample_rate).
    
    Args:
        tensor: PyTorch tensor of shape (channels, samples) or (samples,)
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Tuple of (audio_numpy, sample_rate) matching librosa.load() format
    
    Example:
        >>> audio_tensor = torch.randn(2, 16000)  # stereo audio
        >>> sr = 44100
        >>> audio_np, sr = torch_to_librosa(audio_tensor, sr)
        >>> type(audio_np), audio_np.shape, sr
        (numpy.ndarray, (2, 16000), 44100)
    """
    # Ensure tensor is on CPU and convert to numpy
    audio_np = tensor.cpu().numpy()
    
    # Handle channel dimension (librosa expects channels first)
    if audio_np.ndim == 1:
        # Mono audio (samples,) -> (1, samples)
        audio_np = audio_np[np.newaxis, :]
    elif audio_np.ndim == 2 and audio_np.shape[0] > audio_np.shape[1]:
        # Potential shape mismatch (samples, channels) -> (channels, samples)
        audio_np = audio_np.T
    
    return audio_np, sample_rate
def save_enhanced_audio(
    audio: torch.Tensor, 
    sample_rate: int, 
    output_path: str
) -> None:
    """
    Save enhanced audio tensor to file with validation checks.
    
    Args:
        audio: Enhanced audio tensor (shape: [channels, samples])
        sample_rate: Audio sample rate in Hz
        output_path: Path to save the enhanced audio
        
    Raises:
        ValueError: If audio tensor is invalid or path is not writable
    """
    print(f"\n[SAVE] Attempting to save audio to {output_path}")
    
    # Validate tensor dimensions
    if audio.ndim != 2:
        raise ValueError(f"Audio tensor must be 2D (got {audio.ndim}D)")
    
    # Check output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        torchaudio.save(output_path, audio, sample_rate)
        print(f"[SUCCESS] Audio saved to {output_path}")
        print(f"  - Duration: {audio.shape[1]/sample_rate:.2f}s")
        print(f"  - Channels: {audio.shape[0]}")
        print(f"  - Sample rate: {sample_rate}Hz")
    except Exception as e:
        raise IOError(f"Failed to save audio: {str(e)}")
def detect_audio_input(
    input_data: Union[str, torch.Tensor, np.ndarray, tuple, bytes, io.BytesIO], 
    sr: int = None
) -> Tuple[np.ndarray, int]:
    """
    Universal audio input detector with enhanced debugging.
    
    Args:
        input_data: Audio input (file path, tensor, array, librosa tuple, bytes)
        sr: Sample rate (required for raw arrays/tensors)
        
    Returns:
        Tuple of (audio_array, sample_rate) where:
        - audio_array is shaped (channels, samples)
        - sample_rate is in Hz
        
    Raises:
        ValueError: For unsupported formats or invalid inputs
    """
    # print("\n[DETECT] Audio input detection started") # Debug
    # print(f"[INPUT] Type: {type(input_data)}") # Debug
    
    # Case 1: File path input
    if isinstance(input_data, str):
        # print(f"[FILE] Path detected: {input_data}") # Debug
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Audio file not found: {input_data}")
        
        # Attempt loading with fallback libraries
        loaders = [
            ("torchaudio", lambda: torchaudio.load(input_data, normalize=True)),
            ("librosa", lambda: (librosa.load(input_data, sr=sr, mono=False)[0], sr)),
            ("soundfile", lambda: (sf.read(input_data)[0].T, sf.read(input_data)[1]))
        ]
        
        for loader_name, loader_func in loaders:
            try:
                # print(f"[LOAD] Attempting with {loader_name}...") # Debug
                audio, sr = loader_func() 
                # print(f"[SUCCESS] Loaded with {loader_name}") # Debug
                # print(f"  - Shape: {audio.shape}") # Debug
                # print(f"  - Sample rate: {sr}Hz") # Debug
                
                # Ensure proper shape (channels, samples)
                if audio.ndim == 1:
                    audio = audio[np.newaxis, :]
                elif audio.ndim == 2 and loader_name == "soundfile":
                    audio = audio.T
                    
                return audio, sr
            except Exception as e:
                print(f"[WARNING] {loader_name} failed: {str(e)}")
                continue
                
        raise ValueError("All audio loaders failed for file input")

    # Case 2: PyTorch tensor input
    elif isinstance(input_data, torch.Tensor):
        # print("[TENSOR] PyTorch tensor detected") # Debug
        if sr is None:
            raise ValueError("Sample rate must be provided for tensor input")
            
        audio = input_data.numpy()
        # print(f"  - Original shape: {audio.shape}") # Debug
        
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
            print("  - Added channel dimension")
            
        return audio, sr

    # Case 3: NumPy array input
    elif isinstance(input_data, np.ndarray):
        # print("[ARRAY] NumPy array detected") # Debug
        if sr is None:
            raise ValueError("Sample rate must be provided for array input")
            
        print(f"  - Original shape: {input_data.shape}")
        
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]
            # print("  - Added channel dimension") # Debug
            
        return input_data, sr

    # Case 4: Librosa-style tuple (audio, sample_rate)
    elif isinstance(input_data, tuple) and len(input_data) == 2:
        # print("[LIBROSA] Librosa-style tuple detected") # Debug
        audio, sr = input_data
        
        if not isinstance(audio, np.ndarray):
            raise ValueError("Librosa tuple must contain numpy array")
            
        # print(f"  - Original shape: {audio.shape}") # Debug
        
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
            # print("  - Added channel dimension") # Debug
            
        return audio, sr

    # Case 5: PyAudio stream (bytes)
    elif isinstance(input_data, (bytes, io.BytesIO)):
        # print("[PYAUDIO] Byte stream detected")
        if sr is None:
            raise ValueError("Sample rate must be provided for byte stream")
            
        try:
            # Convert bytes to numpy array (16-bit PCM assumed)
            audio = np.frombuffer(input_data, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            audio = audio[np.newaxis, :]  # Add channel dimension
            # print("  - Converted bytes to float32 tensor") # Debug
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to decode byte stream: {str(e)}")

    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")
def enhance_audio_tensor(
    audio: torch.Tensor, 
    sample_rate: int,
    noise_reduce: bool = True,
    effects: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Enhanced audio processing pipeline with configurable steps.
    
    Args:
        audio: Input audio tensor (shape: [channels, samples])
        sample_rate: Audio sample rate in Hz
        noise_reduce: Whether to apply noise reduction
        effects: Whether to apply audio effects
        
    Returns:
        Tuple of (enhanced_audio, sample_rate)
    """
    # print("\n[ENHANCE] Starting audio enhancement") # Debug
    # print(f"  - Input shape: {audio.shape}") # Debug
    # print(f"  - Sample rate: {sample_rate}Hz") # Debug
    
    audio_np = audio.numpy()
    
    # Noise reduction stage
    if noise_reduce:
        # print("[PROCESS] Applying noise reduction...") # Debug
        try:
            audio_np = nr.reduce_noise(
                y=audio_np,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.75
            )
            # print("  - Noise reduction complete") # Debug
        except Exception as e:
            print(f"[WARNING] Noise reduction failed: {str(e)}")

    # Audio effects stage
    if effects:
        # print("[PROCESS] Applying audio effects chain...") # Debug
        try:
            board = Pedalboard([
                NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
                Compressor(threshold_db=-16, ratio=2.5),
                LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
                Gain(gain_db=10)
            ])
            audio_np = board(audio_np, sample_rate)
            # print("  - Effects processing complete")  # Debug
        except Exception as e:
            print(f"[WARNING] Effects processing failed: {str(e)}")

    # Convert back to tensor
    enhanced_audio = torch.from_numpy(audio_np)
    # print("[SUCCESS] Enhancement complete") # Debug
    # print(f"  - Output shape: {enhanced_audio.shape}") # Debug
    
    return enhanced_audio, sample_rate
def audio_to_tensor(
    input_data: Union[str, torch.Tensor, np.ndarray, tuple, bytes, io.BytesIO], 
    sr: int = None,
    device: str = "cpu",
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Robust audio to tensor converter with detailed logging.
    
    Args:
        input_data: Audio input (multiple formats supported)
        sr: Sample rate (required for some formats)
        device: Target device ("cpu" or "cuda")
        normalize: Whether to normalize to [-1, 1]
        verbose: Enable detailed logging
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    def log(msg):
        if verbose:
            print(f"[CONVERT] {msg}")

    log("Starting conversion process")
    log(f"Input type: {type(input_data)}")
    
    try:
        # Standardize input format
        audio_np, sr = detect_audio_input(input_data, sr)
        log(f"Standardized to numpy array (shape: {audio_np.shape}, SR: {sr}Hz)")

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_np).to(device)
        log(f"Converted to {device} tensor (dtype: {audio_tensor.dtype})")

        # Normalization
        if normalize:
            if audio_tensor.is_floating_point():
                max_val = torch.max(torch.abs(audio_tensor))
                if max_val > 1.0:
                    audio_tensor = audio_tensor / max_val
                    log(f"Normalized to [-1, 1] (max before: {max_val:.4f})")
                else:
                    log("Already in [-1, 1] range, skipping normalization")
            else:
                log("Integer tensor detected, skipping normalization")

        log("Conversion successful")
        return audio_tensor, sr
        
    except Exception as e:
        log(f"Conversion failed: {str(e)}")
        raise
def process_audio_input(
    input_data: Union[str, torch.Tensor, np.ndarray, tuple, bytes, io.BytesIO],
    sr: int = None,
    device: str = "cpu",
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Universal audio input processor that automatically handles conversion to tensor.
    
    Args:
        input_data: Audio input (file path, tensor, array, librosa tuple, bytes)
        sr: Sample rate (required for non-file inputs)
        device: Target device ("cpu" or "cuda")
        normalize: Whether to normalize audio to [-1, 1]
        verbose: Enable debug printing
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    def debug_print(msg):
        if verbose:
            print(f"[DEBUG] {msg}")
    
    debug_print(f"Input type detected: {type(input_data)}")
    
    # Case 1: Already a PyTorch tensor
    if isinstance(input_data, torch.Tensor):
        debug_print("Input is already a PyTorch tensor")
        audio_tensor = input_data.to(device)
        
        # Validate tensor shape
        if audio_tensor.ndim == 1:
            debug_print("Adding channel dimension to mono audio")
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, N)
        elif audio_tensor.ndim > 2:
            raise ValueError(f"Invalid tensor shape: {audio_tensor.shape}. Expected (channels, samples)")
        
        # Normalize if needed
        if normalize and audio_tensor.is_floating_point():
            max_val = torch.max(torch.abs(audio_tensor))
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val
                debug_print(f"Normalized tensor to [-1, 1]. Max before: {max_val:.4f}")
        
        if sr is None:
            raise ValueError("Sample rate must be provided for tensor input")
            
        return audio_tensor, sr
    
    # Case 2: Not a tensor - convert using audio_to_tensor
    else:
        debug_print("Input is not a PyTorch tensor - converting...")
        try:
            return audio_to_tensor(
                input_data=input_data,
                sr=sr,
                device=device,
                normalize=normalize,
                verbose=verbose
            )
        except Exception as e:
            raise ValueError(f"Failed to convert input to tensor: {str(e)}")
def play_tensor_audio(
    audio_tensor: torch.Tensor, 
    sample_rate: int,
    device: int = None,
    blocking: bool = False,
    normalize: bool = True
):
    """
    Enhanced audio playback function with device selection.
    
    Args:
        audio_tensor: Audio tensor (shape: [channels, samples])
        sample_rate: Playback sample rate in Hz
        device: Output device ID (None for default)
        blocking: Whether to wait for playback completion
        normalize: Whether to normalize before playback
    """
    print("\n[PLAYBACK] Preparing audio playback")
    
    # Convert to numpy
    audio_np = audio_tensor.cpu().numpy()
    print(f"  - Input shape: {audio_np.shape}")
    
    # Handle mono/stereo
    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=0)
        print("  - Expanded mono to stereo")
    elif audio_np.ndim > 2:
        raise ValueError("Audio must be 1D (mono) or 2D (stereo)")

    # Normalize if requested
    if normalize:
        max_val = np.max(np.abs(audio_np))
        if max_val > 1.0:
            audio_np = audio_np / max_val
            print(f"  - Normalized (max before: {max_val:.4f})")
    
    # Device info
    if device is not None:
        dev_info = sd.query_devices(device)
        print(f"  - Output device: {dev_info['name']}")

    # Play audio
    try:
        print("  - Starting playback...")
        sd.play(
            audio_np.T,  # sounddevice expects (samples, channels)
            samplerate=sample_rate,
            device=device,
            blocking=blocking
        )
        
        if not blocking:
            print("  - Playback started (non-blocking)")
        else:
            print("  - Playback completed")
    except Exception as e:
        raise RuntimeError(f"Playback failed: {str(e)}")
    
def process_audio(input_audio, sr):
    if torch.is_tensor(input_audio):
        pass
    else:
        # Process audio
        input_audio, sr = process_audio_input(input_audio, sr)
    enhanced_audio, sr = enhance_audio_tensor(input_audio, sr)
    return enhanced_audio,sr

def main(input_audio, sr, output_audio_path):
    """
    Enhanced audio processing pipeline with complete error handling
    
    Args:
        input_audio: Input audio (file path, numpy array, or torch tensor)
        sr: Sample rate (required if input is array/tensor)
        output_audio_path: Path to save enhanced audio
        
    Returns:
        Tuple of (enhanced_audio_tensor, sample_rate) if successful
        None if processing fails
    """
    print(f"\n[MAIN] Starting audio processing pipeline")
    print(f"  - Output will be saved to: {output_audio_path}")
    
    try:
        # Stage 1: Input Conversion
        if not torch.is_tensor(input_audio):
            print("[STAGE 1] Converting input to tensor...")
            input_audio, sr = process_audio_input(input_audio, sr)
            print(f"  - Converted shape: {input_audio.shape}, SR: {sr}Hz")
        else:
            print("[STAGE 1] Input is already a tensor")
            print(f"  - Input shape: {input_audio.shape}, SR: {sr}Hz")
        
        # Stage 2: Audio Enhancement
        print("[STAGE 2] Enhancing audio...")
        enhanced_audio, sr = enhance_audio_tensor(input_audio, sr)
        print(f"  - Enhanced shape: {enhanced_audio.shape}, SR: {sr}Hz")
        
        # Stage 3: Saving
        print("[STAGE 3] Saving enhanced audio...")
        output_audio_path = r"enhanced_audio.wav"
            # Check output directory exists
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

        torchaudio.save(output_audio_path,  enhanced_audio, sample_rate=sr)
        
        print("[SUCCESS] Pipeline completed successfully")
        return enhanced_audio, sr
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        print("Returning None to indicate failure")
        return None
    
if __name__ == "__main__":
    import sys
    import os
    data_path = os.path.abspath(os.path.join(os.getcwd(), '.', 'data'))
    audio_folder=data_path+r"\audio"
    input_audio_path = r'.\audio_1.mp3'
    try:
            # Load audio from other modules
            print("\n[STAGE 1] Loading audio...")
            input_audio, sr = librosa.load(input_audio_path, sr=None, mono=False)
            # input_audio, sr = torchaudio.load(input_audio_path)
            print(f"  - Loaded audio: {input_audio.shape}, {sr}Hz")
            # stage 2 checking audio
            if torch.is_tensor(input_audio):
                pass
            else:
                # Process audio
                input_audio, sr = process_audio_input(input_audio, sr)
            # Play original
            print("\n[STAGE 2] Playing original audio...")
            play_tensor_audio(input_audio, sr,blocking=True)
            
            # Enhance audio
            print("\n[STAGE 3] Enhancing audio...")
            enhanced_audio, sr = enhance_audio_tensor(input_audio, sr)
            print(f"  - Enhanced audio: {enhanced_audio.shape}, {sr}Hz")
            
            # Save enhanced
            print("\n[STAGE 4] Saving enhanced audio...")
            enhace_audio_path = audio_folder+r"\enhanced_audio.wav"
            save_enhanced_audio(enhanced_audio, sr, enhace_audio_path)
            
            # Play enhanced
            print("\n[STAGE 5] Playing enhanced audio...")
            play_tensor_audio(enhanced_audio, sr, blocking=True)
            
            print("\n[SUCCESS] Pipeline completed successfully")
    except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {str(e)}")
            raise