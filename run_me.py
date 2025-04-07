from main import Main
import soundfile as sf
import torchaudio
import librosa

audio_path = '2764-36616-0000.flac'
wav, sample_rate = librosa.load(audio_path,sr=160000)

main = Main()
wav = main.clone_audio(wav, use_vocoder=True)

sf.write("output.wav", wav, sample_rate)