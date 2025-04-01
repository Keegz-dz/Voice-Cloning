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

# Minimum number of frames required for a valid sample (analogous to partials_n_frames)
MIN_FRAMES = 160


def process_speaker(speaker_dir, processed_root, pbar):
    """
    Processes all .flac files for a single speaker directory.
    Saves each processed mel spectrogram as a .npy file and logs them in _sources.txt.
    """
    proc_dir = processed_root / speaker_dir.name
    proc_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    flac_files = list(speaker_dir.rglob('*.flac'))
    
    for flac_file in flac_files:
        try:
            waveform, orig_sr = librosa.load(flac_file, sr=None, mono=True)
            if waveform is None or len(waveform) == 0:
                pbar.write(f"Warning: Skipping empty/invalid file: {flac_file}")
                pbar.update(1)
                continue

            # Preprocess and convert to mel spectrogram
            frames = preprocess_audio(torch.tensor(waveform), orig_sr)
            if frames.shape[0] < MIN_FRAMES:
                pbar.write(f"Skipping {flac_file} (spectrogram too short).")
                pbar.update(1)
                continue

            # Create a unique file name (e.g., frames_1.npy, frames_2.npy, ...)
            file_index = len(sources) + 1
            frames_fname = f"frames_{file_index}.npy"
            np.save(proc_dir / frames_fname, frames)
            sources.append(f"{frames_fname},{flac_file.name}\n")
        except Exception as e:
            pbar.write(f"Error processing {flac_file} for speaker {speaker_dir.name}: {e}")
        finally:
            # Update the progress bar for every file attempted.
            pbar.update(1)
    
    if sources:
        with open(proc_dir / '_sources.txt', 'w') as f:
            f.writelines(sources)
    return len(sources)

def preprocess(raw_data_root: str = 'datasets/LibriSpeech/train-clean-100',
               processed_data_root: str = 'data/processed_data',
               skip_existing: bool = True,
               max_workers: int = None):
    """
    Preprocesses audio data from LibriSpeech train-clean-100, converting raw FLAC files 
    to processed mel spectrogram numpy arrays. Each speaker gets a subfolder with a _sources.txt
    log of source filenames. Uses threading to speed up processing.
    
    Args:
        raw_data_root: Path to the raw LibriSpeech train-clean-100 data directory.
        processed_data_root: Output path for the processed data.
        skip_existing: If True, skips speakers whose processed data (signified by _sources.txt) already exists.
        max_workers: Maximum number of worker threads (defaults to os.cpu_count()).
    """
    raw_root = Path(raw_data_root).resolve()
    processed_root = Path(processed_data_root).resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_root}")

    processed_root.mkdir(parents=True, exist_ok=True)

    print("Starting preprocessing...")
    print(f"  Raw data source:      {raw_root}")
    print(f"  Processed data target: {processed_root}")

    all_speaker_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
    total_speakers_found = len(all_speaker_dirs)
    print(f"Found {total_speakers_found} speaker directories.")

    speakers_to_process = []
    skipped_speaker_count = 0

    if skip_existing:
        print("Scanning for existing data...")
        for speaker_dir in all_speaker_dirs:
            proc_dir = processed_root / speaker_dir.name
            if (proc_dir / '_sources.txt').exists():
                skipped_speaker_count += 1
            else:
                speakers_to_process.append(speaker_dir)
        print(f"-> Skipping {skipped_speaker_count} previously processed speakers.")
    else:
        speakers_to_process = all_speaker_dirs

    if not speakers_to_process:
        print("No new speakers require processing. Exiting.")
        return

    # Count total files to process across all speakers
    total_files_to_process = sum([
        len(list(speaker_dir.rglob('*.flac')))
        for speaker_dir in speakers_to_process
    ])

    if total_files_to_process == 0:
        print("No .flac files found in the selected speakers. Exiting.")
        return

    print(f"Total files to process: {total_files_to_process}")

    # Create a global progress bar
    pbar = tqdm(total=total_files_to_process, desc="Processing files", unit="file")

    # Use ThreadPoolExecutor to process speakers concurrently
    processed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_speaker, speaker_dir, processed_root, pbar): speaker_dir
            for speaker_dir in speakers_to_process
        }
        for future in as_completed(futures):
            speaker = futures[future]
            try:
                processed_count += future.result()
            except Exception as e:
                pbar.write(f"Error processing speaker {speaker.name}: {e}")

    pbar.close()
    print("Preprocessing complete.")
    print(f"Total processed files: {processed_count}")

if __name__ == "__main__":
    preprocess()