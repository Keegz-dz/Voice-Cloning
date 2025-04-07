# ==============================
# Audio Processing Configs 
# ==============================

# Mel-filterbank Configuration (for spectral analysis)
mel_window_length = 25      # Window length for mel spectrogram extraction (ms)
mel_window_step = 10         # Window step size for mel spectrogram extraction (ms)
mel_n_channels = 40          # Number of frequency bands in mel spectrogram

# Core Audio Parameters
sampling_rate = 16000        # Audio sampling rate in Hz
hop_length = int(sampling_rate * mel_window_length / 1000)  # Samples between STFT columns

# Spectrogram Frame Configuration
partials_n_frames = 160      # Number of frames for partial utterances (1600 ms)
inference_n_frames = 80      # Number of frames used during inference (800 ms)


# ==============================
# Voice Activity Detection 
# ==============================

vad_window_length = 30       # Analysis window length for VAD (ms)
vad_moving_average_width = 8  # Smoothing window size for VAD decisions
vad_max_silence_length = 6    # Max allowed consecutive silent frames

# Audio Normalization
audio_norm_target_dBFS = -30  # Target dBFS for audio volume normalization


# ==============================
# Speaker Embedding Model 
# ==============================

# Network Architecture
model_hidden_size = 256      # Size of hidden layers in speaker encoder
model_embedding_size = 256   # Dimension of speaker embeddings
model_num_layers = 3         # Number of LSTM layers in speaker encoder

# Training Parameters
learning_rate_init = 1e-4    # Initial learning rate
speakers_per_batch = 40      # Number of speakers per training batch
utterances_per_speaker = 10  # Number of utterances per speaker in batch


# ==============================
# Tacotron2 & Vocoder Configs 
# ==============================

# Shared Signal Processing Parameters
sample_rate = 16000          # Audio sample rate (must match across components)
n_fft = 800                  # FFT points for STFT
num_mels = 80                # Number of mel bands
hop_size = 200               # Frame shift in samples (12.5ms for 16kHz)
win_size = 800               # Window length in samples (50ms for 16kHz)
fmin = 55                    # Minimum frequency for mel bands
min_level_db = -100          # Reference minimum for dB normalization
ref_level_db = 20            # Reference level for dB normalization
max_abs_value = 4.0          # Maximum absolute value for waveform clipping
preemphasis = 0.97           # Pre-emphasis filter coefficient
preemphasize = True          # Enable pre-emphasis filtering

# Aliases for compatibility
hop_length = hop_size
win_length = win_size
mel_max_abs_value = max_abs_value
apply_preemphasis = preemphasize

# Audio Quantization
bits = 9                     # Bit depth for mu-law encoding
mu_law = True                # Enable mu-law compression


# ==============================
# Tacotron2 TTS Config 
# ==============================

# Network Architecture
tts_embed_dims = 512         # Embedding dimension for text input
tts_encoder_dims = 256       # Encoder convolution channels
tts_decoder_dims = 128       # Decoder LSTM dimensions
tts_postnet_dims = 512       # Postnet convolution channels
tts_encoder_K = 5            # Encoder convolution kernel size
tts_lstm_dims = 1024         # Encoder LSTM dimensions
tts_postnet_K = 5            # Postnet convolution kernel size
tts_num_highways = 4         # Number of highway networks
tts_dropout = 0.5            # Dropout probability

# Text Processing
tts_cleaner_names = ["english_cleaners"]  # Text normalization modules
tts_stop_threshold = -3.4     # Stopping threshold for decoder (in log domain)

# Training Schedule
tts_schedule = [(2, 1e-3, 20_000, 12),   # (reduction factor, learning rate, steps, batch size)
                (2, 5e-4, 40_000, 12),
                (2, 2e-4, 80_000, 12),
                (2, 1e-4, 160_000, 12),
                (2, 3e-5, 320_000, 12),
                (2, 1e-5, 640_000, 12)]

tts_clip_grad_norm = 1.0      # Gradient clipping threshold
tts_eval_interval = 500       # Steps between evaluation/sample generation
tts_eval_num_samples = 1      # Number of samples to generate during evaluation


# ==============================
# Data Preprocessing 
# ==============================

max_mel_frames = 900          # Maximum allowed mel frames per sample
rescale = True                # Enable spectrogram rescaling
rescaling_max = 0.9           # Maximum value for rescaling
synthesis_batch_size = 16     # Batch size for vocoder preprocessing

# Griffin-Lim Parameters (for spectrogram inversion)
signal_normalization = True   # Normalize audio signals
power = 1.5                   # Exponent for magnitude spectrogram inversion
griffin_lim_iters = 60        # Iterations for Griffin-Lim algorithm

# Advanced Audio Processing
fmax = 7600                   # Maximum mel frequency (<= Nyquist frequency)
allow_clipping_in_normalization = True  # Allow clipping during normalization
clip_mels_length = True       # Truncate long mel spectrograms
use_lws = False               # Use Local Weighted Sums for phase reconstruction
symmetric_mels = True         # Use symmetric range for mel values
trim_silence = True           # Trim leading/trailing silence from audio


speaker_embedding_size = 256  # Dimension of speaker embeddings
silence_min_duration_split = 0.4  # Minimum silence duration for splitting (sec)
utterance_min_duration = 1.6  # Minimum utterance duration to retain (sec)


# ==============================
# Vocoder Config 
# ==============================

# Architecture Parameters
voc_mode = 'RAW'              # Output mode: 'RAW' (softmax) or 'MOL' (mixture of logistics)
voc_upsample_factors = (5, 5, 8)  # Upsampling factors (must multiply to hop_length)
voc_rnn_dims = 512            # RNN layer dimensions
voc_fc_dims = 512             # Fully-connected layer dimensions
voc_compute_dims = 128        # Computation layer dimensions
voc_res_out_dims = 128        # Residual block output dimensions
voc_res_blocks = 10           # Number of residual blocks

# Training Configuration
voc_batch_size = 100          # Training batch size
voc_lr = 1e-4                 # Learning rate
voc_gen_at_checkpoint = 5     # Samples to generate per checkpoint
voc_pad = 2                   # Input padding for temporal context
voc_seq_len = hop_length * 5  # Input sequence length (must be multiple of hop_length)

# Synthesis Parameters
voc_gen_batched = True        # Enable batched generation for speed
voc_target = 8000             # Target samples per batch entry
voc_overlap = 400             # Overlap samples between batches