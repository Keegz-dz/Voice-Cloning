from fatchord_version import WaveRNN
import params as p
import torch

class Vocoder():
    def __init__(self):
        self.model = None
        self.device = None

    def load_model(self, weights_fpath, verbose=True):     
        if verbose:
            print("Building Wave-RNN")

        self.model = WaveRNN(
            rnn_dims=p.voc_rnn_dims,
            fc_dims=p.voc_fc_dims,
            bits=p.bits,
            pad=p.voc_pad,
            upsample_factors=p.voc_upsample_factors,
            feat_dims=p.num_mels,
            compute_dims=p.voc_compute_dims,
            res_out_dims=p.voc_res_out_dims,
            res_blocks=p.voc_res_blocks,
            hop_length=p.hop_length,
            sample_rate=p.sample_rate,
            mode=p.voc_mode
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        if verbose:
            print("Loading model weights at %s" % weights_fpath)
        checkpoint = torch.load(weights_fpath, self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()


    def is_loaded(self):
        return self.model is not None


    def infer_waveform(self, mel, normalize=True,  batched=True, target=8000, overlap=800, 
                    progress_callback=None):
        """
        Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
        that of the synthesizer!)
        
        :param normalize:  
        :param batched: 
        :param target: 
        :param overlap: 
        :return: 
        """
        if self.model is None:
            raise Exception("Please load Wave-RNN in memory before using it")
        
        if normalize:
            mel = mel / p.mel_max_abs_value
        mel = torch.from_numpy(mel[None, ...])
        wav = self.model.generate(mel, batched, target, overlap, p.mu_law, progress_callback)
        return wav
