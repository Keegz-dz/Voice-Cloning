from main import Main
import soundfile as sf
import librosa

audio_path = r"datasets/LibriSpeech/train-clean-100/26/495/26-495-0000.flac" 
wav, sample_rate = librosa.load(audio_path, sr = 16000)

main = Main()
# !Always give a text argument or else it will default to Whisper which might break the code on some windows devices.
intended_text_1 = "Last weekend, I went to the zoo with my family. We saw lions, elephants, and monkeys. The birds were colorful and sang beautiful songs. It was exciting to see so many animals in one place."
intended_text_2 = "The sky is blue and birds are flying."

wav = main.clone_audio(wav, use_vocoder=True, text = intended_text_1)

sf.write("output1.wav", wav, sample_rate)

