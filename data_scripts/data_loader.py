# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License 
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express 
# or implied. See the License for the specific language governing permissions and limitations 
# under the License.

import random
import numpy as np
from pathlib import Path
from typing import List
from torch.utils.data import Dataset, DataLoader

# Constant defining the number of frames in each partial utterance.
partials_n_frames = 160  # Adjust as required by your model

class RandomCycler:
    """
    Cycles through items in a sequence in a constrained random order.
    Guarantees that over consecutive queries each item is returned roughly evenly.
    """
    def __init__(self, source: List):
        if not source:
            raise Exception("Can't create RandomCycler from an empty collection")
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count: int) -> List:
        """
        Returns a list of items sampled in random order.
        
        :param count: Number of items to sample.
        :return: List of sampled items.
        """
        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(random.sample(self.all_items, len(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if not self.next_items:
                self.next_items = random.sample(self.all_items, len(self.all_items))
        return out
    
    def __next__(self):
        return self.sample(1)[0]

class Utterance:
    """
    Represents an utterance with paths to its frame data and audio.
    """
    def __init__(self, frames_fpath: Path, wave_fpath: str):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self) -> np.ndarray:
        """Loads and returns the frame data from disk."""
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames: int):
        """
        Crops the frame data into a partial utterance of n_frames.
        
        :param n_frames: Number of frames for the partial utterance.
        :return: Tuple (partial_frames, (start, end)) where partial_frames is the cropped array.
        """
        frames = self.get_frames()
        if frames.shape[0] == n_frames:
            start = 0
        else:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)

class Speaker:
    """
    Holds the collection of utterances for a speaker.
    """
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        
    def _load_utterances(self):
        """
        Loads the speaker's utterances based on the _sources.txt file.
        """
        with self.root.joinpath("_sources.txt").open("r") as sources_file:
            sources = [line.strip().split(",") for line in sources_file if line.strip()]
        # Map frames file to wave file path
        sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sources}
        self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count: int, n_frames: int):
        """
        Samples a batch of partial utterances from the speaker's utterances.
        
        :param count: Number of partial utterances to sample.
        :param n_frames: Number of frames per partial utterance.
        :return: List of tuples (utterance, partial_frames, (start, end)).
        """
        if self.utterances is None:
            self._load_utterances()
        utterances = self.utterance_cycler.sample(count)
        return [(u,) + u.random_partial(n_frames) for u in utterances]

class SpeakerBatch:
    """
    Aggregates partial utterances from a list of speakers into a batch.
    """
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        self.speakers = speakers
        # Sample partial utterances from each speaker.
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        # Combine the partial frames into a single NumPy array for model input.
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])

class SpeakerVerificationDataset(Dataset):
    """
    Dataset that provides a continuous stream of speakers.
    
    Input: A directory containing speaker subdirectories.
    Each speaker subdirectory must include a "_sources.txt" file listing utterance files.
    
    Output: Each dataset item is a Speaker instance.
    """
    def __init__(self, datasets_root: Path):
        self.root = datasets_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if not speaker_dirs:
            raise Exception("No speakers found. Ensure the directory contains speaker folders.")
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        # Simulate an "infinite" dataset.
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
    def get_logs(self):
        """
        Returns concatenated logs from any .txt files in the dataset directory.
        """
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += log_file.read()
        return log_string

class SpeakerVerificationDataLoader(DataLoader):
    """
    DataLoader that creates batches of speakers and aggregates their partial utterances.
    
    Input: A SpeakerVerificationDataset.
    Output: A SpeakerBatch object containing a NumPy array of partial utterance frames.
    """
    def __init__(self, dataset, speakers_per_batch: int, utterances_per_speaker: int, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker
        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
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
        Collates a list of speakers into a single SpeakerBatch.
        """
        return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames)
