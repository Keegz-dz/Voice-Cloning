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
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.current_step = 0  # Track training progress
        self.warmup_steps = 1  # Critical warmup period
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Dynamic margin parameters
        self.register_buffer('effective_s', torch.tensor(0.0))
        self.register_buffer('effective_m', torch.tensor(0.0))
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, input, label):
        # Warmup scheduling
        if self.training:
            progress = min(1.0, self.current_step / self.warmup_steps)
            # Update buffers in-place with scalar values
            self.effective_s.fill_(self.s * progress)
            self.effective_m.fill_(self.m * progress)
            self.current_step += 1
        else:
            self.effective_s.fill_(self.s)
            self.effective_m.fill_(self.m)

        x_norm = F.normalize(input, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        cosine = F.linear(x_norm, w_norm)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-5, 1-1e-5))  # Safer clamping
        
        # Margin adjustment
        phi = cosine * math.cos(self.effective_m) - sine * math.sin(self.effective_m)
        
        if not self.easy_margin:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm * progress)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.effective_s
        return logits

class SpeechEncoderV5(nn.Module):
    def __init__(self, device, loss_device, num_speakers, arc_s=30.0, arc_m=0.30):  # Reduced initial margin
        super(SpeechEncoderV5, self).__init__()
        self.loss_device = loss_device
        
        # Frontend remains unchanged
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.gn1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, mel_n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.gn2 = nn.GroupNorm(4, mel_n_channels)
        self.relu = nn.ReLU().to(device)
        
        # Transformer with gradient checkpointing
        encoder_layer = nn.TransformerEncoderLayer(d_model=mel_n_channels, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=model_num_layers)
        
        self.linear = nn.Linear(mel_n_channels, model_embedding_size)
        
        # Enhanced initialization for ArcFace
        self.arc_margin = ArcMarginProduct(
            model_embedding_size, num_speakers, 
            s=arc_s, m=arc_m
        ).to(loss_device)
        
        # Label smoothing for better convergence
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(loss_device)
        
        # Weight initialization with lower variance
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)  # Tighter initialization
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def do_gradient_ops(self):
        # Adaptive gradient scaling
        for param in self.arc_margin.parameters():
            if param.grad is not None:
                param.grad *= 0.05  # Scale down classifier gradients
                
        clip_grad_norm_(self.parameters(), 5.0)  # Tighter clipping

    def forward(self, utterances):
        x = utterances.unsqueeze(1)
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = torch.mean(x, dim=-1)
        x = x.transpose(1, 2)
        
        # Transformer with gradient checkpointing
        x = torch.utils.checkpoint.checkpoint(
            self.transformer, 
            x.transpose(0, 1),
            use_reentrant=False
        ).transpose(0, 1)
        
        embeds_raw = self.relu(self.linear(x[:, -1, :]))
        embeds = F.normalize(embeds_raw, p=2, dim=1)
        return embeds

    def loss(self, embeds):
        spk_bs, utt_bs, emb_size = embeds.shape
        embeds_flat = embeds.reshape(-1, emb_size).to(self.loss_device)
        
        # Dynamic label generation
        target = torch.repeat_interleave(
            torch.arange(spk_bs, device=self.loss_device),
            utt_bs
        )
        
        logits = self.arc_margin(embeds_flat, target)
        loss = self.loss_fn(logits, target)
        
        # EER computation using pairwise cosine similarity
        with torch.no_grad():
            # Normalize embeddings
            embeds_norm = F.normalize(embeds_flat, p=2, dim=1)
            
            # Create labels for pairwise comparisons (1 if same speaker, 0 otherwise)
            labels = target.unsqueeze(0) == target.unsqueeze(1)
            labels = labels.float().cpu().numpy()
            
            # Compute cosine similarity matrix
            similarity_matrix = torch.mm(embeds_norm, embeds_norm.T).cpu().numpy()
            
            # Flatten matrices and remove diagonal (self-comparisons)
            triu_indices = np.triu_indices(len(similarity_matrix), k=1)
            scores = similarity_matrix[triu_indices]
            labels = labels[triu_indices]
            
            # Compute EER
            fpr, tpr, _ = roc_curve(labels, scores)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        return loss, eer