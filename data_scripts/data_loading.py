from pathlib import Path
from .data_loader import (
    SpeakerVerificationDataset,
    SpeakerVerificationDataLoader,
    partials_n_frames
)

def load_data(processed_root: str = 'data/processed_data',
              speakers_per_batch: int = 2,
              utterances_per_speaker: int = 2,
              num_workers: int = 0):
    """
    Convenience function to load the speaker verification dataset and create a dataloader.
    
    This function simplifies the process of setting up the data pipeline by handling
    the creation of both the dataset and dataloader objects with sensible defaults.
    
    Args:
        processed_root: Path to the directory containing preprocessed speaker data folders
        speakers_per_batch: Number of different speakers to include in each batch
        utterances_per_speaker: Number of utterances to sample from each speaker
        num_workers: Number of worker processes for parallel data loading
    
    Returns:
        tuple: (dataset, loader)
            - dataset: The SpeakerVerificationDataset instance
            - loader: The SpeakerVerificationDataLoader instance
            
    Example:
        >>> dataset, loader = load_data(
        ...     processed_root='path/to/data',
        ...     speakers_per_batch=4,
        ...     utterances_per_speaker=5
        ... )
        >>> for batch in loader:
        ...     # Process the batch
        ...     pass
    """
    # Convert the path string to a resolved Path object
    processed_path = Path(processed_root).resolve()
    
    # Create the dataset
    dataset = SpeakerVerificationDataset(processed_path)
    
    # Create the dataloader with the specified parameters
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch=speakers_per_batch,
        utterances_per_speaker=utterances_per_speaker,
        num_workers=num_workers
    )
    
    return dataset, loader

# Export public API elements
__all__ = ['load_data', 'partials_n_frames']