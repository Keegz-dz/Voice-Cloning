import os
from pathlib import Path
import numpy as np
import librosa
import torch
from tqdm import tqdm

import sys
sys.path.append('.')
from data_preprocessing import *

# Minimum number of frames required for a valid sample (analogous to partials_n_frames)
MIN_FRAMES = 160


def preprocess(raw_data_root: str = 'datasets/LibriSpeech/train-clean-100',
               processed_data_root: str = 'data/processed_data',
               skip_existing: bool = True):
    """
    Preprocesses audio data from LibriSpeech train-clean-100, converting raw FLAC files 
    to processed mel spectrogram numpy arrays. Each speaker gets a subfolder with a _sources.txt
    log of source filenames.

    Args:
        raw_data_root: Path to the raw LibriSpeech train-clean-100 data directory.
        processed_data_root: Output path for the processed data.
        skip_existing: If True, skips speakers whose processed data (signified by _sources.txt) already exists.
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
        for speaker_dir in tqdm(all_speaker_dirs, desc="Scanning speakers"):
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

    # Count total files to process (using all .flac files per speaker)
    total_files_to_process = sum([
        len(list(speaker_dir.rglob('*.flac')))
        for speaker_dir in speakers_to_process
    ])

    if total_files_to_process == 0:
         print("No .flac files found in the selected speakers. Exiting.")
         return

    print(f"Total files to process: {total_files_to_process}")

    with tqdm(total=total_files_to_process, desc="Processing files", unit="file") as pbar:
        for speaker_dir in speakers_to_process:
            proc_dir = processed_root / speaker_dir.name
            proc_dir.mkdir(parents=True, exist_ok=True)
            sources = []

            # Process all .flac files for this speaker
            flac_files = list(speaker_dir.rglob('*.flac'))
            for flac_file in flac_files:
                try:
                    waveform, orig_sr = librosa.load(flac_file, sr=None, mono=True)
                    if waveform is None or len(waveform) == 0:
                        pbar.write(f"Warning: Skipping empty/invalid file: {flac_file}")
                        continue

                    # Preprocess and convert to mel spectrogram
                    frames = preprocess_audio(torch.tensor(waveform), orig_sr)
                    if frames.shape[0] < MIN_FRAMES:
                        pbar.write(f"Skipping {flac_file} (spectrogram too short).")
                        continue

                    # Create a unique file name (e.g., frames_1.npy, frames_2.npy, ...)
                    file_index = len(sources) + 1
                    frames_fname = f"frames_{file_index}.npy"
                    np.save(proc_dir / frames_fname, frames)
                    sources.append(f"{frames_fname},{flac_file.name}\n")

                    pbar.update(1)
                    pbar.set_postfix(speaker=speaker_dir.name, file=f"{flac_file.name[:15]}...")
                except Exception as e:
                    pbar.write(f"Error processing {flac_file} for speaker {speaker_dir.name}: {e}")

            # Write the sources file if some files were processed for this speaker
            if sources:
                with open(proc_dir / '_sources.txt', 'w') as f:
                    f.writelines(sources)

    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess()