from main import Main
import soundfile as sf
import librosa

audio_path = r"C:\Users\anike\Downloads\harvar.flac"
wav, sample_rate = librosa.load(audio_path, sr = 16000)

main = Main()
wav = main.clone_audio(wav, use_vocoder=True)

sf.write("output.wav", wav, sample_rate)

