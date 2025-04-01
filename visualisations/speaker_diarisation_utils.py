""" This module provides a function to run an interactive animation for speaker diarization.
The animation displays similarity curves over time for each speaker and synchronizes with
audio playback.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from time import sleep, perf_counter as timer
from sys import stderr                                      # standard error stream

import numpy as np
import sys
sys.path.append("temp")

from params import sampling_rate


# Matplotlib’s Colour configuration
_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_my_colors = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=float) / 255   # convert these to the 0–1 range that Matplotlib expects for colours


def play_wav(wav, blocking=True):
    """
    Play a waveform using the sounddevice library.
    
    Parameters:
        wav (np.ndarray): The audio waveform.
        blocking (bool): If True, the function will block until audio playback is complete.
    """
    try:
        import sounddevice as sd
        # !Pad waveform with zeros to compensate for sounddevice bug -- truncates the last 0.5 seconds of audio.
        wav = np.concatenate((wav, np.zeros(sampling_rate // 2)))
        sd.play(wav, sampling_rate, blocking=blocking)  # waits until playback is finished
        
    except Exception as e:
        print("Failed to play audio: %s" % repr(e))
        
        
def interactive_diarization(similarity_dict, wav, wav_splits, x_crop=5, show_time=False):
    """
    The function displays a real-time plot updating similarity curves for each speaker.
    At every frame, it computes the current speaker based on similarity thresholds and updates
    the plot annotation accordingly. Audio playback is synchronized to provide a real-time feel.

    Parameters:
        similarity_dict (dict): A dictionary mapping speaker names to 1D NumPy arrays of similarity scores (one score per frame).
        wav (np.ndarray): The full audio waveform.
        wav_splits (list): List of slice objects representing partial segments of the waveform.
                           Used to compute time centers for the embedding slices.
        x_crop (int, optional): Number of seconds (or equivalent frames) to display on the x-axis at once.
                                Default is 5.
        show_time (bool, optional): when true, shows time labels on the x-axis
    """
    fig, ax = plt.subplots()

    # Initialize empty line plots for each speaker based on keys in the similarity dictionary
    lines = [ax.plot([], [], label=name)[0] for name in similarity_dict.keys()]

    # Create a text object that will display the current speaker prediction
    text = ax.text(0, 0, "", fontsize=10)


    def init():
        """
        Initialize the animation plot.
        """
        ax.set_ylim(0.4, 1)  # ?Set expected similarity score range
        ax.set_ylabel("Similarity")
        if show_time:
            ax.set_xlabel("Time (seconds)")
        else:
            ax.set_xticks([])
        ax.set_title("Diarization")
        ax.legend(loc="lower right")
        
        # Return all artists that will be updated during the animation
        return lines + [text]

    # Compute center times (in seconds) for each wav split to serve as the time axis.
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    
    # The frame rate is determined by taking the inverse of the time difference between the first two time centers. This gives the number of frames (or embedding slices) per second.
    rate = 1 / (times[1] - times[0])
    crop_range = int(np.round(x_crop * rate))
    ticks = np.arange(0, len(wav_splits), rate)
    
    # Record reference time for synchronization with audio playback
    ref_time = timer()


    def update(i):
        """
        Update function called for each frame of the animation.

        Parameters:
            i (int): Current frame index.

        Returns:
            list: List of updated line objects and text object for blitting.
        """
        # Define the x-axis crop so that the current frame i is centered in the display window
        crop = (max(i - crop_range // 2, 0), i + crop_range // 2)
        ax.set_xlim(i - crop_range // 2, crop[1])
        if show_time:
            # Adjust x-axis tick labels to display corresponding time values
            crop_ticks = ticks[(crop[0] <= ticks) * (ticks <= crop[1])]
            ax.set_xticks(crop_ticks)
            ax.set_xticklabels(np.round(crop_ticks / rate).astype(np.int))

        # Extract similarity values for the current frame from all speakers
        similarities = [s[i] for s in similarity_dict.values()]
        
        # Identify the speaker with the highest similarity score at the current frame
        best = np.argmax(similarities)
        name, similarity = list(similarity_dict.keys())[best], similarities[best]
        
        # Determine the label and color based on similarity thresholds
        if similarity > 0.75:
            message = "Speaker: %s (confident)" % name
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][best]
        elif similarity > 0.65:
            message = "Speaker: %s (uncertain)" % name
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][best]
        else:
            message = "Unknown/No speaker"
            color = "black"

        # Update the text annotation with the current prediction
        text.set_text(message)
        text.set_c(color)
        text.set_position((i, 0.96))

        # Update line data for each speaker's similarity curve within the current crop
        for line, (name, sim_values) in zip(lines, similarity_dict.items()):
            line.set_data(range(crop[0], i + 1), sim_values[crop[0]:i + 1])

        # Synchronize animation with audio playback using a high-resolution timer
        current_time = timer() - ref_time
        if current_time < times[i]:
            sleep(times[i] - current_time)
        elif current_time - 0.2 > times[i]:
            print("Animation is delayed further than 200ms!", file=stderr)

        return lines + [text]

    # Create and run the animation using FuncAnimation
    ani = FuncAnimation(fig, update, frames=len(wav_splits), init_func=init,
                        blit=not show_time, repeat=False, interval=1)
    
    # Start asynchronous audio playback
    play_wav(wav, blocking=False)
    plt.show()