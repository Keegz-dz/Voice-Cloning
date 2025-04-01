import os
from pathlib import Path
import numpy as np
import librosa
import torch
from tqdm import tqdm
from .audio_preprocessing import preprocess_audio


def preprocess(raw_data_root: str = 'datasets/LibriSpeech/train-clean-100',
              processed_data_root: str = 'data/processed_data',
              max_files_per_speaker: int = 3,
              skip_existing: bool = True):
    """
    Preprocesses audio data, converting raw FLAC files to processed numpy arrays,
    with optimized skipping of already processed speakers and clearer logging.

    Args:
        raw_data_root: Path to the raw LibriSpeech data directory.
        processed_data_root: Output path for the processed data.
        max_files_per_speaker: Maximum number of audio files to process per speaker.
        skip_existing: If True, skips speakers whose processed data (signified by
                       '_sources.txt') already exists.
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
    print(f"Found {total_speakers_found} total speaker directories.")

    speakers_to_process = []
    skipped_speaker_count = 0

    if skip_existing:
        print("Scanning for existing data to determine processing scope...")
        # Use tqdm for visual feedback during the speaker scan
        for speaker_dir in tqdm(all_speaker_dirs, desc="Scanning speakers"):
            proc_dir = processed_root / speaker_dir.name
            if (proc_dir / '_sources.txt').exists():
                skipped_speaker_count += 1
            else:
                speakers_to_process.append(speaker_dir)
        print(f"-> Skipping {skipped_speaker_count} previously processed speakers.")
    else:
        print("-> Skipping check disabled. Will attempt to process all found speakers.")
        speakers_to_process = all_speaker_dirs

    speakers_to_process_count = len(speakers_to_process)

    # Exit early if no speakers need processing
    if speakers_to_process_count == 0:
        # Adjust message based on whether skipping was enabled
        if skip_existing:
             print("-> No new speakers require processing.")
        else:
             print("-> No speakers found to process.")
        print("Preprocessing complete.")
        return

    print(f"-> Preparing to process {speakers_to_process_count} speakers.")

    # Calculate total files accurately based only on speakers to be processed
    total_files_to_process = sum([
        # Sort files for consistent selection, then limit
        len(sorted(list(speaker_dir.rglob('*.flac')))[:max_files_per_speaker])
        for speaker_dir in speakers_to_process
    ])

    # Exit early if no actual files are found for the selected speakers
    if total_files_to_process == 0:
         print(f"-> No .flac files found within the limits ({max_files_per_speaker} per speaker) for the selected {speakers_to_process_count} speakers.")
         print("Preprocessing complete.")
         return

    print(f"   Total files to process: {total_files_to_process}")

    # Main processing loop using tqdm for file progress
    with tqdm(total=total_files_to_process, desc="Processing files", unit="file") as pbar:
        for speaker_dir in speakers_to_process:
            proc_dir = processed_root / speaker_dir.name
            proc_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

            sources = []
            # Get files, sort for consistency, then limit
            flac_files = sorted(list(speaker_dir.rglob('*.flac')))[:max_files_per_speaker]

            for i, flac_file in enumerate(flac_files):
                try:
                    waveform, orig_sr = librosa.load(flac_file, sr=None, mono=True)

                    # Basic check for valid audio loading
                    if waveform is None or len(waveform) == 0:
                        pbar.write(f"Warning: Skipping empty/invalid file: {flac_file}")
                        continue # Skip processing this file

                    # Assuming preprocess_audio handles resampling and feature extraction
                    processed_wav = preprocess_audio(torch.tensor(waveform), orig_sr)

                    frames_fname = f"frames_{i+1}.npy"
                    np.save(proc_dir / frames_fname, processed_wav)
                    sources.append(f"{frames_fname},{flac_file.name}\n")

                    pbar.update(1)
                    # Update progress bar postfix with current speaker and file name snippet
                    pbar.set_postfix(speaker=speaker_dir.name, file=f"{flac_file.name[:15]}...")

                except Exception as e:
                     # Log error and continue processing other files/speakers
                     # This prevents one bad file from stopping the entire process
                     pbar.write(f"Error processing file {flac_file} for speaker {speaker_dir.name}: {e}")

            # Write the sources file only if some files were successfully processed for this speaker
            if sources:
                 with open(proc_dir / '_sources.txt', 'w') as f:
                    f.writelines(sources)

    print("Preprocessing complete.")