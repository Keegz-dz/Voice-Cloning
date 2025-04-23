import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
import numpy as np
from params import *

class SpeechEncoderV3(nn.Module):
    def __init__(self, device, loss_device):
        super(SpeechEncoderV3, self).__init__()
        self.loss_device = loss_device
        
        # Convolutional Front-End
        # Assume the input spectrogram shape is (batch_size, n_frames, n_channels)
        # and we add a channel dimension to treat it as a 2D input
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, 
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # GroupNorm can stabilizes training across channels
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=mel_n_channels, 
                               kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=mel_n_channels)
        self.relu = nn.ReLU().to(device)
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=mel_n_channels, nhead=8, batch_first= True),
            num_layers=model_num_layers,
            norm=nn.LayerNorm(mel_n_channels)  # Standard LayerNorm inside Transformer.
        )
        
        # Linear projection to the final embedding dimension.
        self.linear = nn.Linear(in_features=mel_n_channels, out_features=model_embedding_size)
        
        # Similarity parameters
        self.similarity_weight = nn.Parameter(torch.tensor([10.], device=loss_device))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.], device=loss_device))
        
        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
        # Apply custom weight initialization.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Advanced weight initialization:
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
        # For Transformer layers, the default initialization in PyTorch is typically adequate,
        # but you can add additional rules if desired.

    def do_gradient_ops(self):
        # Scale similarity parameters' gradients.
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
        
        # Clip gradients to stabilize training.
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, utterances):
        """
        Computes speaker embeddings.
        
        :param utterances: Tensor of shape (batch_size, n_frames, n_channels).
        :return: L2-normalized embeddings of shape (batch_size, model_embedding_size).
        """
        # --- CNN Front-End ---
        # Add a channel dimension: (batch_size, 1, n_frames, n_channels)
        x = utterances.unsqueeze(1)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        # After conv layers, x has shape (batch_size, mel_n_channels, new_time, new_freq)
        # Pool over the frequency dimension to reduce it.
        x = torch.mean(x, dim=-1)  # Now shape: (batch_size, mel_n_channels, new_time)
        
        # --- Transformer Front-End ---
        # Transpose to (batch_size, new_time, mel_n_channels)
        x = x.transpose(1, 2)
        # Transformer expects input shape (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Back to (batch_size, seq_len, mel_n_channels)
        # For simplicity, use the last time-step's output.
        embeds_raw = self.relu(self.linear(x[:, -1, :]))
        # L2-normalize the embeddings.
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)
        return embeds

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix between speaker embeddings.
        (Assumes embeds is of shape (speakers_per_batch, utterances_per_speaker, embedding_size))
        """
        embeds = embeds.to(self.loss_device)
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Inclusive centroids: average per speaker.
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids: average excluding current utterance.
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Compute similarity matrix.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker, speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def loss(self, embeds):
        """
        Computes the GE2E-inspired loss and Equal Error Rate (EER) metric.
        
        :param embeds: Tensor of shape (speakers_per_batch, utterances_per_speaker, embedding_size).
        :return: (loss, EER)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)

        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return loss, eer
