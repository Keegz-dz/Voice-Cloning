from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
import numpy as np
from params import *
import math


class ArcMarginProduct(nn.Module):
    """Implements Additive Angular Margin Loss by modifying logits to enforce angular separation between classes.
        Args:
            in_features: size of each input sample
            out_features: size of each output sample (number of classes)
            s: norm of input feature (scale)
            m: margin (in radians)
            easy_margin: whether to use the easy margin
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features      # d, embedding dimension
        self.out_features = out_features    # S, Number of classes (speakers)
        self.s = s                          # Scale factor
        self.m = m                          # Angular margin in radians
        
        # Learnable speaker‐centroid weights: shape (S, d)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))    
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for cos(θ + m) formula
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)         # threshold = cos(π - m)
        self.mm = math.sin(math.pi - m) * m     # offset for hard margin

    """
        Adapted from the ArcFace formulation in:
        Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019).
        Modifications:
        1. Clamping to avoid numerical instability in sine computation.
        2. Integration with speaker-specific batch structure (utterances_per_speaker).
        3. EER computation during training for speaker verification.
    """
    
    def forward(self, input, label):
        """
        Forward pass to produce margin‑enhanced logits.
        
        Args:
            input (Tensor): shape (N, d) where N = S*U total utterances.
            label (LongTensor): shape (N,), values in [0..S−1].
        
        Returns:
            logits (Tensor): shape (N, S), ready for CrossEntropyLoss.
        """
        # Normalize embeddings and speaker‐centroids to unit length
        x_norm = F.normalize(input, p=2, dim=1)         # (N, d)
        w_norm = F.normalize(self.weight, p=2, dim=1)   # (S, d)

        # Compute cosine similarity: cosθ = x̂ · ŵ
        cosine = F.linear(x_norm, w_norm)                       # (N, S)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))    # Compute sinθ for margin formula: sinθ = sqrt(1 − cos²θ)
        phi = cosine * self.cos_m - sine * self.sin_m           # φ = cos(θ + m) = cosθ·cos m − sinθ·sin m 

        # Apply easy or hard margin to avoid instability
        if self.easy_margin:
            # If cosθ < 0, revert to cosθ to avoids flipping signs when θ is large
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # If cosθ < cos(π−m), revert to (cosθ − mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One‑hot encode the true speaker labels
        one_hot = torch.zeros_like(cosine, device=cosine.device)  
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        logits = one_hot * phi + (1.0 - one_hot) * cosine       # Merge: true‐class uses φ, others use original cosine
        logits *= self.s                                        # Scale logits by s for stronger softmax gradients

        return logits


class SpeechEncoderV4(nn.Module):
    def __init__(self, device, loss_device, num_speakers, arc_s=30.0, arc_m=0.50):
        super(SpeechEncoderV4, self).__init__()
        self.loss_device = loss_device
        
        # Convolutional Front-End
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.gn1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, mel_n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.gn2 = nn.GroupNorm(4, mel_n_channels)
        self.relu = nn.ReLU().to(device)
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=mel_n_channels, nhead=8, batch_first=True),
            num_layers=model_num_layers,
            norm=nn.LayerNorm(mel_n_channels)
        )
        
        # Linear projection to embedding
        self.linear = nn.Linear(mel_n_channels, model_embedding_size)
        
        # ArcFace head
        self.arc_margin = ArcMarginProduct(
            in_features=model_embedding_size,
            out_features=num_speakers,
            s=arc_s,
            m=arc_m
        ).to(loss_device)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def do_gradient_ops(self):
        # Clip gradients
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, utterances):
        # utterances: (batch, n_frames, n_channels)
        x = utterances.unsqueeze(1)             # (batch, 1, n_frames, n_channels)
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))  # (batch, mel_n_channels, new_time, new_freq)
        x = torch.mean(x, dim=-1)               # (batch, mel_n_channels, new_time)
        x = x.transpose(1, 2)                   # (batch, new_time, mel_n_channels)
        x = x.transpose(0, 1)                   # (new_time, batch, mel_n_channels)
        x = self.transformer(x)
        x = x.transpose(0, 1)                   # (batch, new_time, mel_n_channels)
        embeds_raw = self.relu(self.linear(x[:, -1, :]))
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)
        return embeds

    def loss(self, embeds):
        # embeds: (speakers_per_batch, utterances_per_speaker, emb_size)
        spk_bs, utt_bs, emb_size = embeds.shape
        batch_size = spk_bs * utt_bs
        embeds_flat = embeds.reshape(batch_size, emb_size).to(self.loss_device)

        # Create labels
        ground_truth = np.repeat(np.arange(spk_bs), utt_bs)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)

        # ArcFace logits & loss
        logits = self.arc_margin(embeds_flat, target)
        loss = self.loss_fn(logits, target)

        # Compute EER
        with torch.no_grad():
            labels = np.eye(spk_bs)[ground_truth]
            preds = logits.detach().cpu().numpy()
            fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return loss, eer
