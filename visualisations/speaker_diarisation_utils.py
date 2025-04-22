""" This module provides a function to run an interactive animation for speaker diarization.
The animation displays similarity curves over time for each speaker and synchronizes with
audio playback. The speaker with the highest similarity score is always displayed.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import sys
import os
sys.path.append("temp")

from params import sampling_rate

# Silence Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Matplotlib's Colour configuration
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
        
        # Ensure waveform is float32 and normalized to prevent distortion
        wav_norm = wav.astype(np.float32)
        if wav_norm.max() > 1.0 or wav_norm.min() < -1.0:
            wav_norm = wav_norm / max(abs(wav_norm.max()), abs(wav_norm.min()))
        
        # Add small buffer at the end to avoid truncation
        wav_norm = np.concatenate((wav_norm, np.zeros(sampling_rate // 2, dtype=np.float32)))
        
        # Use a higher buffer size for better playback stability
        sd.play(wav_norm, sampling_rate, blocking=blocking, blocksize=1024)
        
    except Exception as e:
        print(f"Failed to play audio: {repr(e)}")


def interactive_diarization(similarity_dict, wav, wav_splits, x_crop=5, show_time=False):
    """
    Run an interactive diarization visualization with synchronized audio.
    Always shows the speaker with the highest similarity score.
    
    Parameters:
        similarity_dict (dict): A dictionary mapping speaker names to 1D NumPy arrays of similarity scores.
        wav (np.ndarray): The full audio waveform.
        wav_splits (list): List of slice objects representing partial segments of the waveform.
        x_crop (int, optional): Number of seconds to display on the x-axis at once. Default is 5.
        show_time (bool, optional): When true, shows time labels on the x-axis.
    """
    # Set up the figure and axes
    plt.rcParams['figure.figsize'] = (10, 6)
    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    
    # Calculate times for each segment
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    rate = 1 / (times[1] - times[0]) if len(times) > 1 else 1
    crop_range = int(np.round(x_crop * rate))
    
    # Pre-compute ticks for better performance
    ticks = np.arange(0, len(wav_splits), rate)
    tick_labels = np.round(ticks / rate).astype(np.int32)
    
    # Initialize lines for each speaker
    lines = []
    for idx, name in enumerate(similarity_dict.keys()):
        color = _default_colors[idx % len(_default_colors)]
        line, = ax.plot([], [], label=name, color=color, linewidth=2)
        lines.append(line)
    
    # Text for displaying current speaker
    text = ax.text(0, 0, "", fontsize=10, fontweight='bold')
    
    # Confidence indicator text
    confidence_text = ax.text(0, 0.92, "", fontsize=8, alpha=0.7)
    
    # Set up axes properties
    ax.set_ylim(0.4, 1)
    ax.set_ylabel("Similarity", fontsize=12)
    if show_time:
        ax.set_xlabel("Time (seconds)", fontsize=12)
    else:
        ax.set_xticks([])
    ax.set_title("Speaker Diarization", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    # Initialize with all data to ensure lines are visible
    for idx, (name, sim) in enumerate(similarity_dict.items()):
        lines[idx].set_data(range(len(sim)), sim)
    
    # Setting initial viewing window
    ax.set_xlim(0, min(crop_range, len(wav_splits)-1))
    
    def init():
        # Initialize function for animation
        # Keep the initial data to ensure lines are visible from the start
        for line, (name, sim) in zip(lines, similarity_dict.items()):
            line.set_data(range(len(sim)), sim)
        text.set_text("")
        confidence_text.set_text("")
        return lines + [text, confidence_text]
    
    def update(i):
        # Define the x-axis crop window
        crop = (max(i - crop_range // 2, 0), min(i + crop_range // 2, len(wav_splits)-1))
        ax.set_xlim(crop[0], crop[1])
        
        if show_time:
            # Display time labels
            visible_ticks = ticks[(crop[0] <= ticks) & (ticks <= crop[1])]
            visible_labels = tick_labels[(crop[0] <= ticks) & (ticks <= crop[1])]
            
            if len(visible_ticks) > 0:
                ax.set_xticks(visible_ticks)
                ax.set_xticklabels(visible_labels)
        
        # Extract similarity values for current frame
        similarities = [s[i] for s in similarity_dict.values()]
        
        # Identify best speaker (always pick the highest score)
        best = np.argmax(similarities)
        name = list(similarity_dict.keys())[best]
        similarity = similarities[best]
        
        # Calculate the difference between top two scores for confidence assessment
        sorted_similarities = sorted(similarities, reverse=True)
        margin = sorted_similarities[0] - sorted_similarities[1] if len(sorted_similarities) > 1 else 1.0
        
        # Set message - always show the speaker with highest score
        message = f"Speaker: {name}"
        color = _default_colors[best % len(_default_colors)]
        
        # Set confidence indicator text
        if similarity > 0.75:
            confidence = "High confidence"
        elif similarity > 0.65:
            confidence = "Medium confidence"
        else:
            confidence = "Low confidence"
            
        # Add margin info for close calls
        if margin < 0.05 and len(similarities) > 1:
            second_best = np.argsort(similarities)[-2]
            second_name = list(similarity_dict.keys())[second_best]
            confidence += f" (close: {second_name})"
        
        # Update text
        text.set_text(message)
        text.set_color(color)
        text.set_position((i, 0.96))
        
        confidence_text.set_text(confidence)
        confidence_text.set_position((i, 0.92))
        
        return lines + [text, confidence_text]
    
    # Calculate optimal interval based on audio duration and frames
    duration = len(wav) / sampling_rate
    n_frames = len(wav_splits)
    interval = max(1, int(duration * 1000 / n_frames * 0.95))  # 95% of theoretical frame time
    
    # Start audio playback in a separate thread
    audio_thread = threading.Thread(target=play_wav, args=(wav, True))
    audio_thread.daemon = True
    audio_thread.start()
    
    # Create and run animation
    ani = FuncAnimation(
        fig, 
        update, 
        frames=range(len(wav_splits)), 
        init_func=init,
        blit=False,  # Disable blitting for debugging
        repeat=False, 
        interval=interval
    )
    
    plt.tight_layout()
    plt.show()