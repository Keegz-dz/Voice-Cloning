import numpy as np
import librosa
from scipy.stats import kurtosis, skew
import scipy.signal as signal

def calculate_voice_metrics(audio_data, sr):
    """
    Master function that orchestrates the calculation of all voice quality metrics.
    
    This function calls individual metric calculation functions and normalizes all results
    to a 0-100 scale for consistent interpretation. Higher values indicate better quality.
    
    Args:
        audio_data: Audio time series (numpy array)
        sr: Sampling rate of the audio in Hz
        
    Returns:
        Dictionary containing four key voice metrics:
        - Timbre Richness: Measures voice harmonic structure and spectral quality
        - Pitch Stability: Measures consistency and natural variation in pitch
        - Articulation: Measures clarity of phoneme pronunciation
        - Speech Rhythm: Measures naturalness of speech tempo and pausing patterns
    """
    # Input validation
    if audio_data is None or len(audio_data) == 0:
        return None
    
    # Calculate individual metrics
    timbre_richness = calculate_timbre_richness(audio_data, sr)
    pitch_stability = calculate_pitch_stability(audio_data, sr)
    articulation = calculate_articulation_clarity(audio_data, sr)
    speech_rhythm = calculate_speech_rhythm(audio_data, sr)
    
    # Ensure all metrics are properly bounded within 0-100 range
    metrics = {
        "Timbre Richness": min(max(timbre_richness, 0), 100),
        "Pitch Stability": min(max(pitch_stability, 0), 100),
        "Articulation": min(max(articulation, 0), 100),
        "Speech Rhythm": min(max(speech_rhythm, 0), 100)
    }
    
    return metrics


def calculate_timbre_richness(audio_data, sr):
    """
    Analyzes the spectral characteristics of voice to determine timbre richness.
    
    Timbre richness is calculated through a weighted combination of four spectral features:
    1. Spectral centroid: Represents the "brightness" of sound (higher = brighter)
    2. Spectral bandwidth: Represents the spread of frequencies (higher = richer)
    3. Spectral contrast: Represents the distinction between peaks and valleys in spectrum
    4. Spectral flatness: Measures how noise-like vs. tonal the signal is
    
    Args:
        audio_data: Audio time series (numpy array)
        sr: Sampling rate of the audio in Hz
        
    Returns:
        A score between 0-100 where higher values indicate richer timbre
    """
    # Extract four complementary spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr), axis=0)
    spec_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
    
    # CENTROID ANALYSIS: Higher centroid indicates brighter timbre (max 50 points)
    # Normalized against quarter of sampling rate as a reference point
    centroid_factor = np.mean(spec_centroid) / (sr/4)
    centroid_score = min(centroid_factor * 50, 50)
    
    # BANDWIDTH ANALYSIS: Higher bandwidth indicates richer frequency spread (max 25 points)
    bandwidth_factor = np.mean(spec_bandwidth) / (sr/3)
    bandwidth_score = min(bandwidth_factor * 25, 25)
    
    # CONTRAST ANALYSIS: Higher contrast indicates clearer harmonic structure (max 15 points)
    contrast_mean = np.mean(spec_contrast)
    contrast_score = min(contrast_mean * 5, 15)
    
    # FLATNESS ANALYSIS: Lower flatness (less noise-like) is better for speech (max 10 points)
    # Invert so higher score means better (less flat/noisy)
    flatness_factor = 1 - np.mean(spec_flatness)
    flatness_score = flatness_factor * 10
    
    # Combine all spectral metrics into final score
    timbre_score = centroid_score + bandwidth_score + contrast_score + flatness_score
    
    return min(timbre_score, 100)  # Cap at 100


def calculate_pitch_stability(audio_data, sr):
    """
    Analyzes pitch contour to determine stability and natural variation.
    
    This function uses pitch tracking to identify how consistently a speaker maintains
    pitch while allowing for natural variations. Both excessive stability (robotic)
    and excessive variation (unstable) are penalized.
    
    Key pitch metrics calculated:
    1. Coefficient of variation: Normalized measure of pitch variability
    2. Jitter: Cycle-to-cycle pitch variations (micro-instabilities)
    
    Args:
        audio_data: Audio time series (numpy array)
        sr: Sampling rate of the audio in Hz
        
    Returns:
        A score between 0-100 where higher values indicate optimal pitch stability
    """
    try:
        # Track pitch using PYIN algorithm (more accurate than standard YIN)
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, fmin=70, fmax=400)
        
        # Extract prominent pitches with sufficient confidence
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            confidence = magnitudes[index, i]
            
            # Only include reliable pitch estimates
            if confidence > 0.01 and pitch > 0:
                pitch_values.append(pitch)
        
        # Return default if insufficient data
        if len(pitch_values) < 4:
            return 75
        
        # Calculate pitch statistics
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        
        # Calculate coefficient of variation (CV) - normalized standard deviation
        cv = pitch_std / pitch_mean if pitch_mean > 0 else 1.0
        
        # Score based on CV - penalizing both too stable (robotic) and too variable
        # Ideal natural speech CV range: 0.05-0.15
        if cv < 0.03:      # Too stable/robotic
            stability_score = 75
        elif cv < 0.05:    # Slightly too stable
            stability_score = 85
        elif cv < 0.15:    # Ideal natural range
            stability_score = 95
        elif cv < 0.2:     # Slightly unstable
            stability_score = 85
        elif cv < 0.25:    # Moderately unstable
            stability_score = 75
        else:              # Highly unstable
            stability_score = 60
            
        # Calculate jitter (cycle-to-cycle variation)
        diffs = np.abs(np.diff(pitch_values))
        if len(diffs) > 0:
            mean_diff = np.mean(diffs)
            jitter = mean_diff / pitch_mean if pitch_mean > 0 else 1.0
            
            # Convert jitter to score (lower jitter = higher score)
            jitter_score = max(0, min(100, 100 - (jitter * 1000)))
            
            # Weighted combination of stability and jitter scores
            return (stability_score * 0.7) + (jitter_score * 0.3)
        
        return stability_score
        
    except Exception:
        # Return reasonable default if analysis fails
        return 75


def calculate_articulation_clarity(audio_data, sr):
    """
    Analyzes articulation clarity based on onset strength and frequency distribution.
    
    Articulation is assessed through:
    1. Onset detection: Identifies clear syllable/phoneme boundaries
    2. High-frequency energy: Measures consonant clarity (consonants have more high-frequency content)
    
    Args:
        audio_data: Audio time series (numpy array)
        sr: Sampling rate of the audio in Hz
        
    Returns:
        A score between 0-100 where higher values indicate clearer articulation
    """
    # PART 1: Onset strength analysis (60% of score)
    try:
        # Calculate onset envelope and detect onsets
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        # Calculate onset density (onsets per second)
        duration = len(audio_data) / sr
        if duration > 0:
            onset_density = len(onsets) / duration
            
            # Score based on ideal syllable rate (4-7 syllables/sec is optimal)
            if onset_density < 2:      # Too slow/unclear
                onset_score = 70
            elif onset_density < 4:    # Slightly slow
                onset_score = 80
            elif onset_density < 7:    # Ideal range
                onset_score = 95
            elif onset_density < 9:    # Slightly fast
                onset_score = 85
            else:                      # Too rapid
                onset_score = 75
        else:
            onset_score = 75
    except Exception:
        onset_score = 75
    
    # PART 2: Frequency distribution analysis (40% of score)
    try:
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Separate high frequency bands (consonants) from low (vowels)
        upper_bands = mel_spec_db[64:, :]  # Upper half of mel bands
        lower_bands = mel_spec_db[:64, :]  # Lower half of mel bands
        
        # Calculate average energy in each region
        upper_energy = np.mean(upper_bands)
        lower_energy = np.mean(lower_bands)
        
        # Calculate ratio of consonant to vowel energy
        if lower_energy != 0:
            consonant_ratio = (upper_energy / lower_energy) + 1
            consonant_score = min(90, max(60, consonant_ratio * 30))
        else:
            consonant_score = 75
    except Exception:
        consonant_score = 75
    
    # Combine scores with weights (onset is more important)
    articulation_score = (onset_score * 0.6) + (consonant_score * 0.4)
    
    return articulation_score


def calculate_speech_rhythm(audio_data, sr):
    """
    Analyzes speech rhythm based on energy dynamics, pause patterns, and tempo.
    
    Speech rhythm is evaluated through:
    1. Energy variance: Measures dynamic range in speech intensity
    2. Pause analysis: Evaluates duration and distribution of pauses
    3. Tempo regularity: Assesses appropriateness of speech rate
    
    Args:
        audio_data: Audio time series (numpy array)
        sr: Sampling rate of the audio in Hz
        
    Returns:
        A score between 0-100 where higher values indicate more natural rhythm
    """
    # PART 1: Energy dynamics analysis (40% of score)
    energy = librosa.feature.rms(y=audio_data)[0]
    
    if len(energy) > 1:
        # Calculate energy statistics
        energy_var = np.var(energy)
        energy_range = np.max(energy) - np.min(energy)
        
        # Normalize metrics for scoring
        norm_var = min(energy_var * 200, 1.0)
        
        # Score based on normalized variance
        # Natural speech has moderate variance - not monotone, not chaotic
        if norm_var < 0.3:     # Too monotone
            var_score = 70
        elif norm_var < 0.7:   # Good dynamic range
            var_score = 90
        else:                  # Too variable
            var_score = 80
    else:
        var_score = 75
    
    # PART 2: Pause pattern analysis (40% of score)
    try:
        # Detect silent regions (pauses)
        intervals = librosa.effects.split(audio_data, top_db=30)
        
        if len(intervals) > 1:
            # Calculate pause durations between speech segments
            pauses = []
            for i in range(len(intervals)-1):
                pause_length = (intervals[i+1][0] - intervals[i][1]) / sr
                if pause_length > 0.05:  # Minimum meaningful pause threshold
                    pauses.append(pause_length)
            
            if len(pauses) > 0:
                # Calculate pause statistics
                mean_pause = np.mean(pauses)
                pause_std = np.std(pauses)
                
                # Score based on mean pause length
                if mean_pause < 0.2:     # Very short pauses
                    pause_score = 75
                elif mean_pause < 0.5:   # Good average pause length
                    pause_score = 90
                else:                    # Too long pauses
                    pause_score = 70
                
                # Adjust for pause variability
                if len(pauses) > 1 and mean_pause > 0:
                    cv_pause = pause_std / mean_pause
                    if cv_pause < 0.3:     # Too mechanical/regular
                        pause_score -= 10
                    elif cv_pause > 1.0:   # Too irregular
                        pause_score -= 5
            else:
                pause_score = 75
        else:
            pause_score = 70  # No pauses detected
    except Exception:
        pause_score = 75
    
    # PART 3: Tempo analysis (20% of score)
    try:
        # Calculate tempo and beat strength
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # Score based on tempo appropriateness
        if tempo < 80:        # Too slow for natural speech
            tempo_score = 75
        elif tempo < 160:     # Good range for speech
            tempo_score = 90
        else:                 # Too fast
            tempo_score = 80
    except Exception:
        tempo_score = 75
    
    # Combine all rhythm components with appropriate weights
    rhythm_score = (var_score * 0.4) + (pause_score * 0.4) + (tempo_score * 0.2)
    
    return rhythm_score