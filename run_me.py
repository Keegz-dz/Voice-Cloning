from main import Main
import soundfile as sf
import torchaudio

audio_path = "D:\CODING\SpeechEncoder\data\LibriSpeech/train-clean-100/2764/36616/2764-36616-0000.flac"
wav, sample_rate = torchaudio.load(audio_path)

main = Main(wav)
wav = main.clone_audio(use_vocoder=True)

sf.write("output.wav", wav, sample_rate)