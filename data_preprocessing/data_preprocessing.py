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

# Minimum number of frames required for a valid sample.
MIN_FRAMES = 160

# =============================================================================
# FUNCTIONS
# =============================================================================

def process_speaker(speaker_dir: Path, processed_root: Path, pbar: tqdm) -> int:
    """
    Processes all .flac files for a single speaker directory.
    Saves each processed mel spectrogram as a .npy file and logs them in _sources.txt.

    Args:
        speaker_dir (Path): Path to the speaker's directory containing .flac files.
        processed_root (Path): Path to the root directory where processed data will be saved.
        pbar (tqdm): Progress bar instance to track the processing.

    Returns:
        int: The number of processed files for the speaker.
    """
    # Create the output directory for the speaker.
    proc_dir = processed_root / speaker_dir.name
    proc_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    flac_files = list(speaker_dir.rglob('*.flac'))

    for flac_file in flac_files:
        try:
            # Load the audio waveform.
            waveform, orig_sr = librosa.load(flac_file, sr=None, mono=True)
            if waveform is None or len(waveform) == 0:
                pbar.write(f"Warning: Skipping empty/invalid file: {flac_file}")
                pbar.update(1)
                continue

            # Preprocess the waveform.
            processed_wav = preprocess_audio(torch.tensor(waveform), orig_sr)

            # Convert the processed waveform to a mel spectrogram.
            frames = wav_to_mel_spectrogram(processed_wav, TARGET_SAMPLE_RATE)

            # Skip samples with too few frames.
            if frames.shape[0] < MIN_FRAMES:
                pbar.write(f"Skipping {flac_file} (spectrogram too short).")
                pbar.update(1)
                continue

            # Create a unique filename for the mel spectrogram.
            file_index = len(sources) + 1
            frames_fname = f"frames_{file_index}.npy"
            # Save the mel spectrogram as a NumPy array.
            np.save(proc_dir / frames_fname, frames)
            # Log the processed file.
            sources.append(f"{frames_fname},{flac_file.name}\n")
        except Exception as e:
            pbar.write(f"Error processing {flac_file} for speaker {speaker_dir.name}: {e}")
        finally:
            # Update the progress bar.
            pbar.update(1)

    # Write the list of processed files to _sources.txt.
    if sources:
        with open(proc_dir / '_sources.txt', 'w') as f:
            f.writelines(sources)
    return len(sources)


def preprocess(raw_data_root: str = 'datasets/LibriSpeech/train-clean-100',
               processed_data_root: str = 'data/processed_data',
               skip_existing: bool = True,
               max_workers: int = None):
    """
    Preprocesses audio data from a raw dataset (default: LibriSpeech train-clean-100),
    converting raw FLAC files to processed mel spectrogram numpy arrays.
    Each speaker's data is saved in a subfolder within the processed data root,
    along with a '_sources.txt' file logging the source filenames.
    Uses threading to speed up processing.

    Args:
        raw_data_root (str): Path to the root directory of the raw audio dataset.
                                 Defaults to 'datasets/LibriSpeech/train-clean-100'.
        processed_data_root (str): Output path for the processed mel spectrogram data.
                                     Defaults to 'data/processed_data'.
        skip_existing (bool): If True, skips processing for speakers whose processed
                               data (indicated by the presence of '_sources.txt')
                               already exists. Defaults to True.
        max_workers (int, optional): Maximum number of worker threads to use for
                                     parallel processing. Defaults to the number of
                                     CPU cores.
    """
    # Resolve the input and output directory paths.
    raw_root = Path(raw_data_root).resolve()
    processed_root = Path(processed_data_root).resolve()

    # Ensure the raw data root directory exists.
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_root}")

    # Create the processed data root directory.
    processed_root.mkdir(parents=True, exist_ok=True)

    print("Starting preprocessing...")
    print(f"  Raw data source:      {raw_root}")
    print(f"  Processed data target: {processed_root}")

    # Identify all speaker directories.
    all_speaker_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
    total_speakers_found = len(all_speaker_dirs)
    print(f"Found {total_speakers_found} speaker directories.")

    speakers_to_process = []
    skipped_speaker_count = 0

    # Determine which speakers need processing based on the 'skip_existing' flag.
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

    # Exit if no new speakers to process.
    if not speakers_to_process:
        print("No new speakers require processing. Exiting.")
        return

    # Count total files to process.
    total_files_to_process = sum([
        len(list(speaker_dir.rglob('*.flac')))
        for speaker_dir in speakers_to_process
    ])

    # Exit if no .flac files found.
    if total_files_to_process == 0:
        print("No .flac files found in the selected speakers. Exiting.")
        return

    print(f"Total files to process: {total_files_to_process}")

    # Create a global progress bar.
    pbar = tqdm(total=total_files_to_process, desc="Processing files", unit="file")

    # Use ThreadPoolExecutor for parallel processing.
    processed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for processing each speaker.
        futures = {
            executor.submit(process_speaker, speaker_dir, processed_root, pbar): speaker_dir
            for speaker_dir in speakers_to_process
        }
        # Wait for tasks to complete and handle potential errors.
        for future in as_completed(futures):
            speaker = futures[future]
            try:
                processed_count += future.result()
            except Exception as e:
                pbar.write(f"Error processing speaker {speaker.name}: {e}")

    # Close the progress bar.
    pbar.close()
    print("Preprocessing complete.")
    print(f"Total processed files: {processed_count}")

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    preprocess()