# =============================================================================
# SPEECH PROCESSING SYSTEM PARAMETERS
# This file contains all hyperparameters for the audio processing pipeline,
# neural network models, and training configuration.
# =============================================================================

# =============================================================================
# FEATURE EXTRACTION PARAMETERS
# =============================================================================

# Mel-filterbank configuration for speaker embeddings
mel_window_length = 25  # Window size in milliseconds
mel_window_step = 10    # Frame shift in milliseconds
mel_n_channels = 40     # Number of mel bins for speaker recognition

# Audio processing parameters
sampling_rate = 16000   # Audio sampling rate in Hz
# Convert window length from milliseconds to samples
hop_length = int(sampling_rate * mel_window_length / 1000)

# Utterance segmentation parameters
partials_n_frames = 160  # Number of frames in a partial utterance (1600 ms)
inference_n_frames = 80  # Number of frames at inference time (800 ms)

# =============================================================================
# VOICE ACTIVITY DETECTION (VAD) PARAMETERS
# =============================================================================

vad_window_length = 30  # VAD analysis window length in milliseconds
# Smoothing factor - higher values lead to less sensitivity to short silence
vad_moving_average_width = 8
# Maximum consecutive silent frames allowed within a speech segment
vad_max_silence_length = 6

# =============================================================================
# AUDIO PREPROCESSING
# =============================================================================

# Target volume normalization level in decibels Full Scale
audio_norm_target_dBFS = -30

# =============================================================================
# SPEAKER ENCODER MODEL ARCHITECTURE
# =============================================================================

model_hidden_size = 256      # Size of LSTM hidden states
model_embedding_size = 256   # Speaker embedding vector dimension
model_num_layers = 3         # Number of LSTM layers

# =============================================================================
# SPEAKER ENCODER TRAINING PARAMETERS
# =============================================================================

learning_rate_init = 1e-4     # Initial learning rate
speakers_per_batch = 40       # Number of speakers per training batch
utterances_per_speaker = 10   # Number of utterances sampled per speaker in a batch

# =============================================================================
# SIGNAL PROCESSING PARAMETERS (SHARED BETWEEN SYNTHESIZER AND VOCODER)
# =============================================================================

sample_rate = 16000           # Audio sampling rate in Hz
n_fft = 800                   # FFT size
num_mels = 80                 # Number of mel filterbanks for synthesis
hop_size = 200                # Frame shift (12.5 ms at 16kHz)
win_size = 800                # Window size (50 ms at 16kHz)
fmin = 55                     # Minimum frequency for mel filterbanks
min_level_db = -100           # Minimum dB threshold for spectrograms
ref_level_db = 20             # Reference level dB for normalization
max_abs_value = 4.0           # Spectrogram normalization range
preemphasis = 0.97            # Pre-emphasis filter coefficient
preemphasize = True           # Whether to apply pre-emphasis
hop_length = hop_size         # Alias for hop_size for compatibility

# Wavenet vocoder specific parameters
bits = 9                      # Bit depth for quantized audio (mu-law)
mu_law = True                 # Whether to use mu-law quantization

# Redundant assignments for compatibility with different parts of the codebase
sample_rate = sample_rate
n_fft = n_fft
num_mels = num_mels
hop_length = hop_size
win_length = win_size
fmin = fmin
min_level_db = min_level_db
ref_level_db = ref_level_db
mel_max_abs_value = max_abs_value
preemphasis = preemphasis
apply_preemphasis = preemphasize

# =============================================================================
# TACOTRON 2 TEXT-TO-SPEECH (TTS) PARAMETERS
# =============================================================================

# Network dimensions
tts_embed_dims = 512          # Text embedding dimension
tts_encoder_dims = 256        # Encoder LSTM dimensions
tts_decoder_dims = 128        # Decoder LSTM dimensions
tts_postnet_dims = 512        # Postnet convolutional dimensions
tts_encoder_K = 5             # Encoder convolutional filter size
tts_lstm_dims = 1024          # Dimensions of the LSTM pre-decoder
tts_postnet_K = 5             # Postnet convolutional filter size
tts_num_highways = 4          # Number of highway layers
tts_dropout = 0.5             # Dropout rate
tts_cleaner_names = ["english_cleaners"]  # Text normalization methods

# Stopping condition for audio generation
tts_stop_threshold = -3.4     # Threshold for predicted stop token (range: [-4, 4])

# =============================================================================
# TACOTRON TRAINING SCHEDULE
# =============================================================================

# Progressive training schedule (reduction factor, learning rate, step, batch size)
tts_schedule = [(2,  1e-3,  20_000,  12),   # Initial phase
                (2,  5e-4,  40_000,  12),   # Reduction in learning rate as training progresses
                (2,  2e-4,  80_000,  12),   
                (2,  1e-4, 160_000,  12),   # r = reduction factor (# of mel frames
                (2,  3e-5, 320_000,  12),   #     synthesized per decoder iteration)
                (2,  1e-5, 640_000,  12)],  # lr = learning rate

tts_clip_grad_norm = 1.0       # Gradient norm clipping to prevent explosion
tts_eval_interval = 500        # Steps between model evaluations
tts_eval_num_samples = 1       # Number of evaluation samples to generate

# =============================================================================
# DATA PREPROCESSING PARAMETERS
# =============================================================================

max_mel_frames = 900           # Maximum number of mel frames in an utterance
rescale = True                 # Whether to rescale audio
rescaling_max = 0.9            # Maximum amplitude after rescaling
synthesis_batch_size = 16      # Batch size for vocoder preprocessing

# =============================================================================
# MEL VISUALIZATION AND GRIFFIN-LIM PARAMETERS
# =============================================================================

signal_normalization = True    # Whether to normalize mel spectrograms
power = 1.5                    # Power coefficient for Griffin-Lim
griffin_lim_iters = 60         # Number of Griffin-Lim iterations

# =============================================================================
# AUDIO PROCESSING OPTIONS
# =============================================================================

fmax = 7600                               # Maximum frequency for mel filterbanks
allow_clipping_in_normalization = True    # Allow clipping when normalizing signals
clip_mels_length = True                   # Discard samples exceeding max_mel_frames
use_lws = False                           # Local weighted sums for phase recovery
symmetric_mels = True                     # Whether to use symmetric mel range 
trim_silence = True                       # Remove silence from audio samples

# =============================================================================
# SV2TTS (SPEAKER VERIFICATION TO TEXT-TO-SPEECH) PARAMETERS
# =============================================================================

speaker_embedding_size = 256              # Speaker embedding dimension
silence_min_duration_split = 0.4          # Minimum silence duration to split utterances (seconds)
utterance_min_duration = 1.6              # Minimum utterance duration to keep (seconds)

# =============================================================================
# NEURAL VOCODER PARAMETERS
# =============================================================================

voc_mode = 'RAW'                          # 'RAW' (softmax) or 'MOL' (mixture of logistics)
voc_upsample_factors = (5, 5, 8)          # Upsampling factors (must multiply to hop_length)
voc_rnn_dims = 512                        # RNN dimensions in vocoder
voc_fc_dims = 512                         # Fully connected layer dimensions
voc_compute_dims = 128                    # Compute dimensions for residual connections
voc_res_out_dims = 128                    # Output dimensions for residual connections
voc_res_blocks = 10                       # Number of residual blocks

# =============================================================================
# VOCODER TRAINING PARAMETERS
# =============================================================================

voc_batch_size = 100                      # Training batch size
voc_lr = 1e-4                             # Learning rate
voc_gen_at_checkpoint = 5                 # Number of samples to generate at checkpoints
voc_pad = 2                               # Input padding for wider receptive field
voc_seq_len = hop_length * 5              # Sequence length (must be multiple of hop_length)

# =============================================================================
# AUDIO GENERATION / SYNTHESIS PARAMETERS
# =============================================================================

voc_gen_batched = True                    # Whether to use batched generation
voc_target = 8000                         # Target samples in each batch entry
voc_overlap = 400                         # Overlap between consecutive batches