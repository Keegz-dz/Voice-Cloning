import numpy as np
import librosa
from scipy.stats import kurtosis, skew
import scipy.signal as signal

def calculate_voice_metrics(audio_data, sr):
    """
    Calculate actual voice metrics based on audio analysis.
    
    Args:
        audio_data: Audio time series (numpy array)
        sr: Sampling rate of the audio
        
    Returns:
        Dictionary of voice metrics
    """
    # Make sure we have audio data to analyze
    if audio_data is None or len(audio_data) == 0:
        return None
    
    # Calculate timbre richness
    timbre_richness = calculate_timbre_richness(audio_data, sr)
    
    # Calculate pitch stability
    pitch_stability = calculate_pitch_stability(audio_data, sr)
    
    # Calculate articulation clarity
    articulation = calculate_articulation_clarity(audio_data, sr)
    
    # Calculate speech rhythm
    speech_rhythm = calculate_speech_rhythm(audio_data, sr)
    
    # Scale all metrics to 0-100 range (they may be on different scales initially)
    metrics = {
        "Timbre Richness": min(max(timbre_richness, 0), 100),
        "Pitch Stability": min(max(pitch_stability, 0), 100),
        "Articulation": min(max(articulation, 0), 100),
        "Speech Rhythm": min(max(speech_rhythm, 0), 100)
    }
    
    return metrics


def calculate_timbre_richness(audio_data, sr):
    """
    Calculate timbre richness based on spectral features.
    Higher values indicate richer harmonic structure.
    
    Uses spectral centroid, bandwidth, contrast, and flatness.
    """
    # Calculate spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr), axis=0)
    spec_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
    
    # Higher centroid can indicate brighter timbre
    centroid_factor = np.mean(spec_centroid) / (sr/4)  # Normalize by quarter of sample rate
    centroid_score = min(centroid_factor * 50, 50)  # Scale to max 50
    
    # Higher bandwidth indicates more spread across frequencies
    bandwidth_factor = np.mean(spec_bandwidth) / (sr/3)  # Normalize 
    bandwidth_score = min(bandwidth_factor * 25, 25)  # Scale to max 25
    
    # Higher contrast indicates clearer harmonic structure
    contrast_mean = np.mean(spec_contrast)
    contrast_score = min(contrast_mean * 5, 15)  # Scale to max 15
    
    # Lower flatness (less noise-like) is better for speech
    # Invert flatness so higher is better
    flatness_factor = 1 - np.mean(spec_flatness)
    flatness_score = flatness_factor * 10  # Scale to max 10
    
    # Combine scores
    timbre_score = centroid_score + bandwidth_score + contrast_score + flatness_score
    
    # Scale to 0-100
    return min(timbre_score, 100)


def calculate_pitch_stability(audio_data, sr):
    """
    Calculate pitch stability by analyzing pitch contour and variations.
    Higher values indicate more stable pitch with natural variations.
    
    Uses pitch tracking and analyzes the consistency of pitch values.
    """
    # Track pitch (f0)
    try:
        # Use PYIN (more accurate than standard YIN)
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, fmin=70, fmax=400)
        
        # Get the most prominent pitch for each frame
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            confidence = magnitudes[index, i]
            
            # Only keep pitches with sufficient confidence/magnitude
            if confidence > 0.01 and pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) < 4:  # Need enough values for meaningful analysis
            return 75  # Default to average if not enough data
        
        # Calculate stability metrics
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        
        # Calculate coefficient of variation (normalized std)
        # Lower values are more stable but some variation is natural
        if pitch_mean > 0:
            cv = pitch_std / pitch_mean
        else:
            cv = 1.0  # Default if mean is zero
        
        # Calculate stability
        # Ideal CV is around 0.05-0.15 for natural speech
        # Too low (< 0.03) is robotic, too high (> 0.2) is unstable
        if cv < 0.03:  # Too stable (robotic)
            stability_score = 75
        elif cv < 0.05:
            stability_score = 85
        elif cv < 0.15:  # Ideal natural range
            stability_score = 95
        elif cv < 0.2:
            stability_score = 85
        elif cv < 0.25:
            stability_score = 75
        else:  # Too unstable
            stability_score = 60
            
        # Calculate jitter (cycle-to-cycle variation)
        diffs = np.abs(np.diff(pitch_values))
        if len(diffs) > 0:
            mean_diff = np.mean(diffs)
            jitter = mean_diff / pitch_mean if pitch_mean > 0 else 1.0
            
            # Adjust score based on jitter
            # Natural speech has small, controlled jitter
            jitter_score = max(0, min(100, 100 - (jitter * 1000)))
            
            # Combine scores (weighted average)
            return (stability_score * 0.7) + (jitter_score * 0.3)
        
        return stability_score
        
    except Exception:
        # If pitch analysis fails, return a reasonable default
        return None


def calculate_articulation_clarity(audio_data, sr):
    """
    Calculate articulation clarity based on speech rate, onset strength,
    and high-frequency content (consonant clarity).
    
    Higher values indicate clearer pronunciations.
    """
    # Calculate onset strength (indicates clear syllable/phoneme boundaries)
    try:
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        # More onsets typically indicate clearer articulation
        # But normalize by duration
        duration = len(audio_data) / sr
        if duration > 0:
            onset_density = len(onsets) / duration
            # Ideal speech has about 4-7 syllables per second
            if onset_density < 2:  # Too slow/unclear
                onset_score = 70
            elif onset_density < 4:
                onset_score = 80
            elif onset_density < 7:  # Ideal range
                onset_score = 95
            elif onset_density < 9:
                onset_score = 85
            else:  # Too rapid
                onset_score = 75
        else:
            onset_score = 75
    except Exception:
        onset_score = None
    
    # Calculate high-frequency energy (consonant clarity)
    # Use mel spectrogram and focus on upper frequency bands
    try:
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Focus on upper half (consonants have more high-frequency energy)
        upper_bands = mel_spec_db[64:, :]
        lower_bands = mel_spec_db[:64, :]
        
        # Calculate average energy
        upper_energy = np.mean(upper_bands)
        lower_energy = np.mean(lower_bands)
        
        # Calculate ratio (higher means more consonant energy)
        if lower_energy != 0:
            consonant_ratio = (upper_energy / lower_energy) + 1  # Normalize
            
            # Scale to score
            consonant_score = min(90, max(60, consonant_ratio * 30))
        else:
            consonant_score = 75
    except Exception:
        consonant_score = 75
    
    # Combine scores (weighted average)
    articulation_score = (onset_score * 0.6) + (consonant_score * 0.4)
    
    return articulation_score


def calculate_speech_rhythm(audio_data, sr):
    """
    Calculate speech rhythm quality based on energy variance,
    pause patterns, and rhythm regularity.
    
    Higher values indicate more natural rhythm patterns.
    """
    # Calculate RMS energy
    energy = librosa.feature.rms(y=audio_data)[0]
    
    # Calculate energy dynamics (variance and range)
    if len(energy) > 1:
        energy_var = np.var(energy)
        energy_range = np.max(energy) - np.min(energy)
        
        # Calculate normalized metrics
        norm_var = min(energy_var * 200, 1.0)  # Scale variance
        norm_range = min(energy_range * 2, 1.0)  # Scale range
        
        # Natural speech has moderate variance and range
        # Too little = monotone, too much = unnatural
        if norm_var < 0.3:  # Too monotone
            var_score = 70
        elif norm_var < 0.7:  # Good range
            var_score = 90
        else:  # Too variable
            var_score = 80
    else:
        var_score = 75
    
    # Detect pauses (silence regions)
    try:
        # Find silent regions
        intervals = librosa.effects.split(audio_data, top_db=30)
        
        if len(intervals) > 1:
            # Calculate pause durations
            pauses = []
            for i in range(len(intervals)-1):
                pause_length = (intervals[i+1][0] - intervals[i][1]) / sr
                if pause_length > 0.05:  # Only count meaningful pauses
                    pauses.append(pause_length)
            
            if len(pauses) > 0:
                # Calculate pause statistics
                mean_pause = np.mean(pauses)
                pause_std = np.std(pauses)
                
                # Natural speech has pauses of varying lengths
                # But too variable is unnatural
                if mean_pause < 0.2:  # Very short pauses
                    pause_score = 75
                elif mean_pause < 0.5:  # Good average pause length
                    pause_score = 90
                else:  # Too long pauses
                    pause_score = 70
                
                # Adjust for pause variability
                if len(pauses) > 1 and mean_pause > 0:
                    cv_pause = pause_std / mean_pause
                    if cv_pause < 0.3:  # Too regular
                        pause_score -= 10
                    elif cv_pause > 1.0:  # Too irregular
                        pause_score -= 5
            else:
                pause_score = 75
        else:
            pause_score = 70  # No pauses detected
    except Exception:
        pause_score = None
    
    # Tempo regularity (rhythm)
    try:
        # Calculate tempo and beat strength
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # Assess tempo appropriateness (natural speech tempo)
        if tempo < 80:  # Too slow
            tempo_score = 75
        elif tempo < 160:  # Good range for speech
            tempo_score = 90
        else:  # Too fast
            tempo_score = 80
    except Exception:
        tempo_score = None
    
    # Combine scores with weights
    rhythm_score = (var_score * 0.4) + (pause_score * 0.4) + (tempo_score * 0.2)
    
    return rhythm_score