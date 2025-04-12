from pathlib import Path
from .data_loader import (
    SpeakerVerificationDataset,
    SpeakerVerificationDataLoader,
    partials_n_frames  # Exported constant
)

# Expose a simple helper function for loading the dataset and dataloader.
def load_data(processed_root: str = 'data/processed_data',
              speakers_per_batch: int = 2,
              utterances_per_speaker: int = 2,
              num_workers: int = 0):
    """
    Loads the speaker verification dataset and creates a dataloader.
    
    Args:
        processed_root: Path to the directory with preprocessed speaker folders.
        speakers_per_batch: Number of speakers per batch.
        utterances_per_speaker: Number of utterances to sample per speaker.
        num_workers: Number of worker threads for data loading.
    
    Returns:
        dataset: The SpeakerVerificationDataset instance.
        loader: The SpeakerVerificationDataLoader instance.
    """
    processed_path = Path(processed_root).resolve()
    dataset = SpeakerVerificationDataset(processed_path)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch=speakers_per_batch,
        utterances_per_speaker=utterances_per_speaker,
        num_workers=num_workers
    )
    return dataset, loader

# Expose the constant for external use if needed.
__all__ = ['load_data', 'partials_n_frames']
