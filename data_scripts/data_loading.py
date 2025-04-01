from pathlib import Path
from .data_loader import (
    SpeakerVerificationDataset,
    SpeakerVerificationDataLoader,
    partials_n_frames  # Add this export
)

# Expose partials_n_frames at the module level
__all__ = ['load_data', 'partials_n_frames']

def load_data(processed_root: str = 'data/processed_data',
             speakers_per_batch: int = 2,
             utterances_per_speaker: int = 2,
             num_workers: int = 0):
    processed_path = Path(processed_root).resolve()
    dataset = SpeakerVerificationDataset(processed_path)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch=speakers_per_batch,
        utterances_per_speaker=utterances_per_speaker,
        num_workers=num_workers
    )
    return dataset, loader