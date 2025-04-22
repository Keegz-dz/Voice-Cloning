from main import Main
import soundfile as sf
import librosa

audio_path = r"Male.mp3" 

wav, sample_rate = librosa.load(audio_path, sr = 16000)

main = Main()
# !Always give a text argument or else it will default to Whisper which might break the code on some windows devices.
intended_text_1 = "Last weekend, I went to the zoo with my family. We saw lions, elephants, and monkeys. The birds were colorful and sang beautiful songs. It was exciting to see so many animals in one place."
intended_text_2 = "Hello everyone, it's nice to speak with you today. This is a demonstration of voice cloning technology."
intended_text_3 = "The goal of a bandit problem is to maximize the total cumulative reward over a sequence of actions by repeatedly choosing from a set of available actions"

wav = main.clone_audio(wav, use_vocoder=True, text = intended_text_2)

sf.write("intended_text_3.wav", wav, sample_rate)

