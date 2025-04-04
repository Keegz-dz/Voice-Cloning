import torch
import hparams
from tacotron import Tacotron
from symbols import symbols
from pathlib import Path
from typing import Union, List
import numpy as np
import librosa
from speech_encoder_v2 import SpeechEncoderV2
from embed import Embed
from temp import text_to_sequence
import audio

class Synthesizer:
    sample_rate = hparams.sample_rate
    hparams = hparams

    def __init__(self, model_fpath: Path, embeddings, verbose=True):
        """
        The model isn't instantiated and loaded in memory until needed or until load() is called.

        :param model_fpath: path to the trained model file
        :param verbose: if False, prints less information when using the model
        """

        self.model_fpath = model_fpath
        self.verbose = verbose
        self.embeddings = embeddings

        # Check for GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.verbose:
            print("Synthesizer using device:", self.device)

        # Tacotron model will be instantiated later on first use.
        self._model = None

    def is_loaded(self):
        """
        Whether the model is loaded in memory.
        """
        return self._model is not None

    def load(self):
        """
        Instantiates and loads the model given the weights file that was passed in the constructor.
        """
        self._model = Tacotron(embed_dims=hparams.tts_embed_dims,
                               num_chars=len(symbols),
                               encoder_dims=hparams.tts_encoder_dims,
                               decoder_dims=hparams.tts_decoder_dims,
                               n_mels=hparams.num_mels,
                               fft_bins=hparams.num_mels,
                               postnet_dims=hparams.tts_postnet_dims,
                               encoder_K=hparams.tts_encoder_K,
                               lstm_dims=hparams.tts_lstm_dims,
                               postnet_K=hparams.tts_postnet_K,
                               num_highways=hparams.tts_num_highways,
                               dropout=hparams.tts_dropout,
                               stop_threshold=hparams.tts_stop_threshold,
                               speaker_embedding_size=hparams.speaker_embedding_size).to(self.device)

        self._model.load(self.model_fpath)
        self._model.eval()

        if self.verbose:
            print("Loaded synthesizer \"%s\" trained to step %d" % (self.model_fpath.name, self._model.state_dict()["step"]))

    def synthesize_spectrograms(self, texts: List[str], return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256)
        :param return_alignments: if True, a matrix representing the alignments between the
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the
        sequence length of spectrogram i, and possibly the alignments.
        """
        # Load the model on the first request.
        if not self.is_loaded():
            self.load()

        # Preprocess text inputs
        inputs = [text_to_sequence(text.strip(), hparams.tts_cleaner_names) for text in texts]
        # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
        # inputs, lengths = utils.prepare_input_sequence([texts])


        # Batch inputs
        batched_inputs = [inputs[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(inputs), hparams.synthesis_batch_size)]
        batched_embeds = [self.embeddings[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(self.embeddings), hparams.synthesis_batch_size)]
        

        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            if self.verbose:
                print(f"\n| Generating {i}/{len(batched_inputs)}")

            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [self.pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            speaker_embeds = np.stack(batched_embeds[i-1])

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)

            # Inference
            _, mels, alignments = self._model.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            for m in mels:
                # Trim silence from end of each spectrogram
                while np.max(m[:, -1]) < hparams.tts_stop_threshold:
                    m = m[:, :-1]
                specs.append(m)

        if self.verbose:
            print("\n\nDone.\n")
        return (specs, alignments) if return_alignments else specs

    def pad1d(self, x, max_len, pad_value=0):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

    def griffin_lim(self, mel):
        """
        Inverts a mel spectrogram using Griffin-Lim. The mel spectrogram is expected to have been built
        with the same parameters present in hparams.py.
        """
        return audio.inv_mel_spectrogram(mel, hparams)