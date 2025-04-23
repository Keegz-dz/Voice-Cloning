import random
import numpy as np
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

# Global constants
PARTIALS_N_FRAMES = 160  # Number of frames in each partial utterance segment

# Modified work based on original code by Corentin Jemine (https://github.com/CorentinJ/Real-Time-Voice-Cloning)
# The following code is licensed under the MIT License
    
class RandomCycler:
    """
    A utility class that cycles through items in a constrained random order.
    
    This ensures that over consecutive sampling operations, each item appears
    with roughly equal frequency, while still maintaining randomness.
    """
    def __init__(self, source: List):
        """
        Initialize the RandomCycler with a collection of items.
        
        Args:
            source: A collection of items to cycle through
        
        Raises:
            Exception: If the source collection is empty
        """
        if not source:
            raise Exception("Can't create RandomCycler from an empty collection")
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count: int) -> List:
        """
        Sample a specified number of items from the collection.
        
        Args:
            count: Number of items to sample
            
        Returns:
            A list of sampled items
        """
        out = []
        while count > 0:
            # If we need more items than available, sample the entire collection
            if count >= len(self.all_items):
                out.extend(random.sample(self.all_items, len(self.all_items)))
                count -= len(self.all_items)
                continue
                
            # Otherwise, take items from the next_items queue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            
            # If next_items is empty, refill it with a new random permutation
            if not self.next_items:
                self.next_items = random.sample(self.all_items, len(self.all_items))
        return out
    
    def __next__(self):
        """Return a single item when used as an iterator."""
        return self.sample(1)[0]

class Utterance:
    """
    Represents a single speech utterance with its associated frame data and audio path.
    
    This class handles loading and processing of individual utterance data from disk.
    """
    def __init__(self, frames_fpath: Path, wave_fpath: str):
        """
        Initialize an utterance with paths to its data.
        
        Args:
            frames_fpath: Path to the preprocessed frames file (.npy)
            wave_fpath: Path to the original audio file
        """
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self) -> np.ndarray:
        """
        Load and return the preprocessed frame data from disk.
        
        Returns:
            NumPy array containing the frame data
        """
        # Load the frames data from the .npy file
        # For large datasets, memory-mapping could be used: np.load(..., mmap_mode='r')
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract a random segment of specified length from the utterance.
        
        Args:
            n_frames: Number of frames to extract
            
        Returns:
            Tuple containing:
                - The extracted frame segment as numpy array
                - A tuple of (start_index, end_index)
        """
        frames = self.get_frames()
        
        # If the utterance is exactly the right length, use the whole thing
        if frames.shape[0] == n_frames:
            start = 0
        else:
            # Otherwise, pick a random starting point
            start = np.random.randint(0, frames.shape[0] - n_frames)
            
        end = start + n_frames
        return frames[start:end], (start, end)

class Speaker:
    """
    Represents a speaker with multiple utterances.
    
    Handles loading and sampling of utterances from a single speaker.
    """
    def __init__(self, root: Path):
        """
        Initialize a speaker from a directory.
        
        Args:
            root: Path to the speaker's directory
        """
        self.root = root
        self.name = root.name
        self.utterances = None  # Lazy-loaded
        self.utterance_cycler = None
        
    def _load_utterances(self):
        """
        Load all utterances for this speaker from the _sources.txt file.
        
        The file should contain lines with format: frames_filename,original_audio_filename
        """
        with self.root.joinpath("_sources.txt").open("r") as sources_file:
            sources = [line.strip().split(",") for line in sources_file if line.strip()]
            
        # Create a mapping of frames file to original wave file
        sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
        
        # Create Utterance objects for each entry
        self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count: int, n_frames: int):
        """
        Sample random partial utterances from this speaker.
        
        Args:
            count: Number of utterances to sample
            n_frames: Number of frames per partial utterance
            
        Returns:
            List of tuples: (utterance, frames, (start, end))
        """
        # Lazy-load utterances if not already loaded
        if self.utterances is None:
            self._load_utterances()
            
        # Sample utterances using the cycler to ensure even distribution
        utterances = self.utterance_cycler.sample(count)
        
        # Extract a random partial from each utterance
        return [(u,) + u.random_partial(n_frames) for u in utterances]

class SpeakerBatch:
    """
    Aggregates partial utterances from multiple speakers into a single batch.
    
    This class is responsible for organizing utterances in a format suitable
    for the deep learning model.
    """
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        """
        Create a batch from multiple speakers.
        
        Args:
            speakers: List of Speaker objects
            utterances_per_speaker: Number of utterances to sample per speaker
            n_frames: Number of frames per utterance
        """
        self.speakers = speakers
        
        # Sample partial utterances from each speaker
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        
        # Aggregate all frame data into a single NumPy array:
        # Shape: (n_speakers * utterances_per_speaker, n_frames, feature_dim)
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])

class SpeakerVerificationDataset(Dataset):
    """
    Dataset for speaker verification that provides a continuous stream of speakers.
    
    This dataset expects a directory structure where each speaker has their own
    subdirectory containing preprocessed utterance data.
    """
    def __init__(self, datasets_root: Path):
        """
        Initialize the dataset from a root directory.
        
        Args:
            datasets_root: Path to the directory containing speaker subdirectories
            
        Raises:
            Exception: If no speaker directories are found
        """
        self.root = datasets_root
        
        # Find all speaker directories
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if not speaker_dirs:
            raise Exception("No speakers found. Ensure the directory contains speaker folders with _sources.txt.")
            
        # Create Speaker objects for each directory
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        """
        Return the dataset size (simulating an "infinite" dataset).
        """
        return int(1e10)
        
    def __getitem__(self, index):
        """
        Return a speaker based on the given index.
        
        Since we're using a RandomCycler, the index is ignored,
        and we return the next speaker from the cycler.
        
        Args:
            index: Index (ignored)
            
        Returns:
            A Speaker object
        """
        return next(self.speaker_cycler)
    
    def get_logs(self):
        """
        Return concatenated logs from any .txt files in the dataset directory.
        
        Returns:
            String containing the contents of all log files
        """
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += log_file.read()
        return log_string

class SpeakerVerificationDataLoader(DataLoader):
    """
    Custom DataLoader for speaker verification.
    
    This DataLoader creates batches of speakers and aggregates their
    partial utterances into SpeakerBatch objects.
    """
    def __init__(self, dataset, speakers_per_batch: int, utterances_per_speaker: int, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        """
        Initialize the DataLoader.
        
        Args:
            dataset: The SpeakerVerificationDataset
            speakers_per_batch: Number of speakers in each batch
            utterances_per_speaker: Number of utterances to sample per speaker
            Other args: Standard DataLoader parameters
        """
        self.utterances_per_speaker = utterances_per_speaker
        
        # Initialize the parent DataLoader with our custom collate function
        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False,  # Shuffling is handled by RandomCycler
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers: List[Speaker]) -> SpeakerBatch:
        """
        Custom collate function that creates a SpeakerBatch.
        
        Args:
            speakers: List of Speaker objects
            
        Returns:
            A SpeakerBatch containing the sampled utterances
        """
        return SpeakerBatch(speakers, self.utterances_per_speaker, PARTIALS_N_FRAMES)

# Export the constant with a more standard naming convention
partials_n_frames = PARTIALS_N_FRAMES