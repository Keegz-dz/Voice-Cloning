import os
from pathlib import Path
import numpy as np
import librosa
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.append('.')
from data_preprocessing import *

# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum number of frames required for a valid sample
# At 10ms per frame, 160 frames = 1.6 seconds of audio
MIN_FRAMES = 160

# =============================================================================
# FUNCTIONS
# =============================================================================

def process_speaker(speaker_dir: Path, processed_root: Path, pbar: tqdm) -> int:
    """
    Processes all audio files for a single speaker and converts them to mel spectrograms.
    
    Workflow:
    1. For each .flac file in the speaker directory:
       a. Load and preprocess the audio (resample, normalize, trim silences)
       b. Convert to mel spectrogram features
       c. Skip samples that are too short (<160 frames ≈ 1.6 seconds)
       d. Save as .npy files for efficient loading during training
    2. Create a _sources.txt file that maps each .npy file to its source audio
    
    This speaker-based organization facilitates training speaker verification models
    and enables per-speaker normalization if needed.
    
    Args:
        speaker_dir (Path): Directory containing a speaker's audio files
        processed_root (Path): Output directory for processed features
        pbar (tqdm): Progress bar for tracking
        
    Returns:
        int: Number of successfully processed files for this speaker
    """
    # Create the output directory for the speaker
    proc_dir = processed_root / speaker_dir.name
    proc_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    flac_files = list(speaker_dir.rglob('*.flac'))

    for flac_file in flac_files:
        try:
            # Load the audio waveform
            waveform, orig_sr = librosa.load(flac_file, sr=None, mono=True)
            if waveform is None or len(waveform) == 0:
                pbar.write(f"Warning: Skipping empty/invalid file: {flac_file}")
                pbar.update(1)
                continue

            # Apply the full preprocessing pipeline from audio_preprocessing.py
            processed_wav = preprocess_audio(torch.tensor(waveform), orig_sr)

            # Convert the processed waveform to a mel spectrogram
            frames = wav_to_mel_spectrogram(processed_wav, TARGET_SAMPLE_RATE)

            # Skip samples with too few frames (too short for meaningful analysis)
            if frames.shape[0] < MIN_FRAMES:
                pbar.write(f"Skipping {flac_file} (spectrogram too short: {frames.shape[0]} frames)")
                pbar.update(1)
                continue

            # Create a unique filename for the mel spectrogram
            file_index = len(sources) + 1
            frames_fname = f"frames_{file_index}.npy"
            
            # Save the mel spectrogram as a NumPy array for efficient loading during training
            np.save(proc_dir / frames_fname, frames)
            
            # Log the processed file and its source for traceability
            sources.append(f"{frames_fname},{flac_file.name}\n")
            
        except Exception as e:
            pbar.write(f"Error processing {flac_file} for speaker {speaker_dir.name}: {e}")
        finally:
            # Update the progress bar
            pbar.update(1)

    # Write the list of processed files to _sources.txt for data tracking
    if sources:
        with open(proc_dir / '_sources.txt', 'w') as f:
            f.writelines(sources)
    
    return len(sources)


def preprocess(raw_data_root: str = 'datasets/LibriSpeech/train-clean-100',
               processed_data_root: str = 'data/processed_data',
               skip_existing: bool = True,
               max_workers: int = None):
    """
    Master preprocessing pipeline for converting raw audio to mel spectrograms.
    
    Key aspects:
    1. Parallel processing: Uses ThreadPoolExecutor to process speakers concurrently
    2. Resumable: Can skip already processed speakers using skip_existing
    3. Output structure: 
       - Each speaker gets a directory named by their ID
       - Each directory contains:
         a. frames_*.npy files (mel spectrograms)
         b. _sources.txt (mapping between .npy files and source audio)
    
    The resulting dataset structure is optimized for training speaker verification
    or voice conversion models where speaker identity is important.
    
    Args:
        raw_data_root (str): Root directory of LibriSpeech or similar dataset
        processed_data_root (str): Output directory for processed features
        skip_existing (bool): Whether to skip speakers that have already been processed
        max_workers (int): Maximum number of parallel processing threads
    """
    # Resolve the input and output directory paths
    raw_root = Path(raw_data_root).resolve()
    processed_root = Path(processed_data_root).resolve()

    # Verify input directory exists
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_root}")

    # Create the output directory
    processed_root.mkdir(parents=True, exist_ok=True)

    print("Starting preprocessing...")
    print(f"  Raw data source:      {raw_root}")
    print(f"  Processed data target: {processed_root}")

    # Identify all speaker directories
    all_speaker_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
    total_speakers_found = len(all_speaker_dirs)
    print(f"Found {total_speakers_found} speaker directories.")

    speakers_to_process = []
    skipped_speaker_count = 0

    # Handle resuming processing with skip_existing
    if skip_existing:
        print("Scanning for existing data...")
        for speaker_dir in all_speaker_dirs:
            proc_dir = processed_root / speaker_dir.name
            # Check if this speaker has already been processed
            if (proc_dir / '_sources.txt').exists():
                skipped_speaker_count += 1
            else:
                speakers_to_process.append(speaker_dir)
        print(f"-> Skipping {skipped_speaker_count} previously processed speakers.")
    else:
        speakers_to_process = all_speaker_dirs

    # Exit if no new speakers to process
    if not speakers_to_process:
        print("No new speakers require processing. Exiting.")
        return

    # Count total files to process for progress tracking
    total_files_to_process = sum([
        len(list(speaker_dir.rglob('*.flac')))
        for speaker_dir in speakers_to_process
    ])

    # Exit if no .flac files found
    if total_files_to_process == 0:
        print("No .flac files found in the selected speakers. Exiting.")
        return

    print(f"Total files to process: {total_files_to_process}")

    # Create a global progress bar
    pbar = tqdm(total=total_files_to_process, desc="Processing files", unit="file")

    # Process speakers in parallel using ThreadPoolExecutor
    # ThreadPoolExecutor is used (rather than ProcessPoolExecutor) because the task is I/O bound
    processed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for processing each speaker
        futures = {
            executor.submit(process_speaker, speaker_dir, processed_root, pbar): speaker_dir
            for speaker_dir in speakers_to_process
        }
        
        # Wait for tasks to complete and handle potential errors
        for future in as_completed(futures):
            speaker = futures[future]
            try:
                processed_count += future.result()
            except Exception as e:
                pbar.write(f"Error processing speaker {speaker.name}: {e}")

    # Close the progress bar and print summary
    pbar.close()
    print("Preprocessing complete.")
    print(f"Total processed files: {processed_count}")

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # Execute the preprocessing pipeline with default parameters
    preprocess()