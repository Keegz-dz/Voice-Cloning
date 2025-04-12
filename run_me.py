from main import Main
import soundfile as sf
import librosa

audio_path = r"C:\Users\anike\Downloads\spanish_output.wav"
wav, sample_rate = librosa.load(audio_path, sr = 16000)

main = Main()
wav = main.clone_audio(wav, use_vocoder=True, text = "The sky is blue and birds are flying.")

sf.write("output.wav", wav, sample_rate)

