import numpy as np
import torch
from scipy.io import wavfile
from transformers import Wav2Vec2BertForSequenceClassification, AutoFeatureExtractor
from scipy.signal import resample


try:
    from packages.audio_config import TEMP_OUTPUT_WAV, RATE
except ImportError:
    # Default values if the import fails
    TEMP_OUTPUT_WAV = "temp_output.wav"
    RATE = 16000

# Load model and processor once
MODEL_PATH = "pipecat-ai/smart-turn"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()
processor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

def predict_endpoint(audio_array):
    """
    Predict whether an audio segment is complete (turn ended) or incomplete.
    
    Args:
        audio_array (np.array): Audio samples at 16kHz
    
    Returns:
        dict: Prediction result { 'prediction': int, 'probability': float }
    """
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        padding="max_length",
        truncation=True,
        max_length=800,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        completion_prob = probabilities[0, 1].item()
        prediction = int(completion_prob > 0.5)
    
    return {"prediction": prediction, "probability": completion_prob}


def save_audio(audio, filename=TEMP_OUTPUT_WAV):
    """Save the audio to a WAV file."""
    wavfile.write(filename, RATE, (audio * 32767).astype(np.int16))

def predict(audio):
    """Run the prediction endpoint and print results."""
    result = predict_endpoint(audio)
    print(f"Prediction: {'Complete' if result['prediction'] else 'Incomplete'} (Prob: {result['probability']:.4f})")
    return result



def process_audio_file(file_path, chunk_duration=3.0,debug=False):
    """
    Process an audio file by splitting it into chunks (including the last incomplete chunk) 
    and running turn detection.
    Automatically handles sample rate conversion.
    """
    # Load the audio file
    try:
        sample_rate, audio_data = wavfile.read(file_path)
    except Exception as e:
        raise ValueError(f"Error loading audio file: {str(e)}")

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Normalize to float32 (-1.0 to 1.0)
    if np.issubdtype(audio_data.dtype, np.integer):
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    else:
        audio_data = audio_data.astype(np.float32)
        peak = np.max(np.abs(audio_data))
        if peak > 1.0:
            audio_data = audio_data / peak

    # Resample if needed
    if sample_rate != RATE:
        # print(f"Resampling from {sample_rate}Hz to {RATE}Hz...")
        duration = len(audio_data) / sample_rate
        target_length = int(duration * RATE)
        audio_data = resample(audio_data, target_length)
        sample_rate = RATE

    # Calculate chunk size in samples
    chunk_size = int(chunk_duration * sample_rate)
    total_samples = len(audio_data)
    total_chunks = (total_samples + chunk_size - 1) // chunk_size  # Ensures last chunk is included
    if debug:
        pass
        print(f"Debug mode: Chunk size = {chunk_size}, Total samples = {total_samples}, Total chunks = {total_chunks}")
        print(f"\nProcessing: {file_path}")
        print(f"Duration: {total_samples/sample_rate:.2f}s")
        print(f"Sample rate: {sample_rate}Hz")
        print(f"Splitting into {total_chunks} chunks (~{chunk_duration}s each, including last partial chunk)")

    # Process each chunk
    results = []
    for i in range(total_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_samples)  # Prevents going beyond audio length
        chunk = audio_data[start:end]
        
        print(f"\nChunk {i+1}/{total_chunks} ({start/sample_rate:.2f}-{end/sample_rate:.2f}s, {len(chunk)/sample_rate:.2f}s):")
        
        # Get prediction
        result = predict(chunk)
        results.append({
            'chunk_number': i+1,
            'start_time': start/sample_rate,
            'end_time': end/sample_rate,
            'duration': end/sample_rate - start/sample_rate,
            'prediction': result['prediction'],
            'probability': result['probability']
        })
    
    return results
if __name__ == "__main__":
    import os
    import sys
    data_path = os.path.abspath(os.path.join(os.getcwd(), '.'))
    sys.path.append(data_path)
    import warnings
    # Suppress WAV file warnings
    warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
    audio_file = r"data\audio\harvard.wav"  # Change this to your audio file
    results = process_audio_file(audio_file)
    
    # Print summary
    print("\nFinal Results:")
    for i, result in enumerate(results):
        print(f"Chunk {i+1}: {'Complete' if result['prediction'] else 'Incomplete'} "
              f"(Prob: {result['probability']:.4f})")