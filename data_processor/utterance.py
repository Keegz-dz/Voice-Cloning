# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import numpy as np

class Utterance:
    def __init__(self, frames_fpath, wave_fpath):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        frames = self.get_frames()
        if frames.shape[0] == n_frames:
            start = 0
        else:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)