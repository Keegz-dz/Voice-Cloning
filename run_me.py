from main import Main
import soundfile as sf
import librosa

audio_path = r"datasets/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
wav, sample_rate = librosa.load(audio_path, sr = 16000)

main = Main(original_encoder=True)
wav = main.clone_audio(wav, use_vocoder=True, text = "The sky is blue and birds are flying.")

sf.write("output.wav", wav, sample_rate)

