from params import *
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from torch import nn
import numpy as np
import torch


class SpeechEncoderV2(nn.Module):
    """
    Transformer-based speech encoder for speaker verification.
    
    This model implements a speaker verification system using a transformer architecture
    followed by a projection layer to create speaker embeddings. It's designed to learn
    discriminative representations that capture speaker identity while being robust to
    other variations in speech content.
    
    The model is trained using the Generalized End-to-End (GE2E) loss function which
    encourages embeddings from the same speaker to be closer together while pushing
    embeddings from different speakers apart.
    
    Architecture overview:
        1. Input: Mel-spectrogram utterances
        2. Positional encoding added to preserve temporal information
        3. Transformer encoder to process the sequence
        4. Mean pooling across time dimension
        5. Linear projection to embedding dimension
        6. L2-normalization to create unit-length embeddings
    """
    
    def __init__(self, device, loss_device):
        """
        Initialize the speech encoder with transformer architecture.
        
        Args:
            device: The device to run the forward pass on (CPU or GPU)
            loss_device: The device to compute the loss on (typically GPU for efficiency)
        """
        super(SpeechEncoderV2, self).__init__()
        self.loss_device = loss_device
        
        # ===== Architecture Components =====
        
        # Positional encoding: Added to input to preserve temporal information
        # This is a learnable parameter (rather than fixed sinusoidal encoding)
        # Shape: [1, 1, mel_n_channels]
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, mel_n_channels))
        
        # Transformer encoder: Processes the sequence using self-attention
        # - d_model: Dimension of input features (mel_n_channels)
        # - nhead: Number of attention heads (8)
        # - num_layers: Number of transformer layers (from params)
        # - norm: Layer normalization applied after transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=mel_n_channels, nhead=8),
            num_layers=model_num_layers,
            norm=nn.LayerNorm(mel_n_channels),
        )
        
        # Projection layer: Maps transformer output to embedding space
        # Reduces dimensionality from mel_n_channels to model_embedding_size
        self.linear = nn.Linear(in_features=mel_n_channels, out_features=model_embedding_size)
        
        # Activation function: Adds non-linearity to the embedding projection
        self.relu = torch.nn.ReLU().to(device)

        # ===== GE2E Loss Parameters =====
        
        # Learnable parameters for similarity scaling in GE2E loss
        # These control the "sharpness" of the softmax distribution:
        # - similarity_weight: Scales the cosine similarities (initially 10)
        # - similarity_bias: Shifts the cosine similarities (initially -5)
        self.similarity_weight = nn.Parameter(torch.tensor([10.], device=loss_device))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.], device=loss_device))

        # Loss function: Cross entropy for classification of speakers
        # Each utterance should be classified as belonging to its speaker
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)

    def do_gradient_ops(self):
        """
        Apply gradient modifications before the optimization step.
        
        This method performs two important operations:
        1. Gradient scaling: Reduces the gradient magnitude for similarity parameters
           by multiplying by 0.01. This prevents these parameters from changing too
           quickly, which stabilizes training.
        
        2. Gradient clipping: Limits the maximum gradient norm to 3.0 across all
           parameters. This prevents exploding gradients which can destabilize
           training, especially in deep networks like transformers.
           
        Both operations are crucial for stable convergence of the model.
        """
        # Scale down gradients for similarity parameters (0.01 factor)
        # These parameters control the scale of the logits and can cause instability
        # if allowed to change too quickly
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
        
        # Clip gradients to prevent exploding gradients
        # Maximum L2 norm of 3.0 for all parameters combined
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances):
        """
        Compute speaker embeddings from utterance spectrograms.
        
        This forward pass transforms mel-spectrograms into speaker embeddings
        through a sequence of operations:
        1. Add positional encoding to preserve temporal information
        2. Process through transformer layers with self-attention
        3. Average pool across time dimension
        4. Project to embedding dimension with non-linearity
        5. L2-normalize to create unit-length embeddings
        
        Args:
            utterances: Batch of mel-spectrograms with shape (batch_size, n_frames, n_channels)
            where:
                - batch_size: Number of utterances
                - n_frames: Number of time frames (can vary)
                - n_channels: Number of mel frequency channels
            
        Returns:
            L2-normalized speaker embeddings with shape (batch_size, embedding_size)
            These embeddings should cluster by speaker identity in the embedding space.
        """
        # Add positional encoding to the input
        # The broadcasting ensures each position in the sequence gets a unique encoding
        # Shape after addition: (batch_size, n_frames, n_channels)
        utterances = utterances + self.pos_encoder[:, :, :utterances.size(1)]
        
        # Pass through transformer layers
        # First transpose to shape (n_frames, batch_size, n_channels) as transformers
        # expect sequence length as first dimension, then transpose back after processing
        out = self.transformer(utterances.transpose(0, 1)).transpose(0, 1)
        
        # Average pooling over the time dimension (dim=1)
        # This creates a fixed-length representation regardless of input sequence length
        # Then apply projection and ReLU activation
        # Shape after linear: (batch_size, embedding_size)
        embeds_raw = self.relu(self.linear(out.mean(dim=1)))
        
        # L2-normalize the embeddings
        # This ensures all embeddings lie on the unit hypersphere, making
        # cosine similarity calculations simpler (just dot products)
        # The epsilon (1e-5) prevents division by zero for zero-norm vectors
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds
    
    def similarity_matrix(self, embeds):
        """
        Compute the similarity matrix between utterance embeddings and speaker centroids.
        
        This implements the core of the GE2E loss function by calculating:
        1. Inclusive centroids: For each speaker, the mean of all their utterances
        2. Exclusive centroids: For each utterance, the mean of all OTHER utterances from the same speaker
        3. Similarity scores between each utterance and all centroids
        
        Mathematical context:
        - For utterance j of speaker i (e_ji) and speaker k's centroid (c_k):
          * If i=k: Calculate similarity using exclusive centroid (excluding e_ji)
          * If i≠k: Calculate similarity using inclusive centroid (all utterances)
        - This prevents trivial solutions where utterances simply match themselves
        - Final similarity scores are scaled and shifted by learnable parameters
        
        Args:
            embeds: Speaker embeddings with shape (speakers_per_batch, utterances_per_speaker, embedding_size)
            
        Returns:
            Similarity matrix with shape (speakers_per_batch, utterances_per_speaker, speakers_per_batch)
            Contains the scaled cosine similarity between each utterance and each speaker centroid
        """
        # Move embeddings to loss device if necessary (e.g., from CPU to GPU)
        embeds = embeds.to(self.loss_device) 
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # ===== Step 1: Calculate inclusive centroids (1 per speaker) =====
        # These are the mean embeddings for each speaker including ALL utterances
        # Shape: (speakers_per_batch, 1, embedding_size)
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        
        # Clone to ensure proper gradient flow and normalize
        # L2-normalization ensures centroids lie on the unit hypersphere like embeddings
        # Shape after normalization: (speakers_per_batch, 1, embedding_size)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # ===== Step 2: Calculate exclusive centroids (1 per utterance) =====
        # These are the mean embeddings for each speaker EXCLUDING the current utterance
        # Formula: (sum_of_all_utterances - current_utterance) / (utterances_per_speaker - 1)
        # Shape: (speakers_per_batch, utterances_per_speaker, embedding_size)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        
        # Clone and normalize exclusive centroids
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # ===== Step 3: Compute similarity matrix =====
        # Initialize empty similarity matrix
        # Shape: (speakers_per_batch, utterances_per_speaker, speakers_per_batch)
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                speakers_per_batch).to(self.loss_device)
        
        # Create a mask to identify non-diagonal elements (other speakers)
        # This gives 0 on diagonal (same speaker) and 1 elsewhere (different speakers)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=int)
        
        # Calculate similarities for each speaker
        for j in range(speakers_per_batch):
            # Get indices of all other speakers
            mask = np.where(mask_matrix[j])[0]
            
            # Calculate similarity between utterances of OTHER speakers and THIS speaker's centroid
            # For each utterance of other speakers, compute dot product with this speaker's centroid
            # Shape: (other_speakers, utterances_per_speaker)
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            
            # Calculate similarity between utterances of THIS speaker and their EXCLUSIVE centroids
            # For each utterance of this speaker, compute dot product with its exclusive centroid
            # Shape: (utterances_per_speaker)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        # ===== Step 4: Apply learnable scaling and shifting =====
        # Scale and shift similarities using learnable parameters
        # This allows the model to adjust the "sharpness" of the softmax distribution
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        
        return sim_matrix
        
    def loss(self, embeds):
        """
        Compute the Generalized End-to-End (GE2E) loss and Equal Error Rate (EER).
        
        The GE2E loss encourages the following:
        1. Embeddings from the same speaker should be similar to each other
        2. Embeddings from different speakers should be dissimilar
        
        It achieves this by treating the problem as a classification task:
        - Each utterance should be classified as belonging to its true speaker
        - The similarity scores serve as logits for this classification
        
        Additionally, this method computes the Equal Error Rate (EER) as an
        evaluation metric (not used for backpropagation).
        
        Args:
            embeds: Speaker embeddings with shape (speakers_per_batch, utterances_per_speaker, embedding_size)
            
        Returns:
            Tuple containing:
                - The GE2E loss value (scalar tensor)
                - The Equal Error Rate (EER) for this batch (float)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # ===== Step 1: Compute similarity matrix =====
        sim_matrix = self.similarity_matrix(embeds)
        
        # Reshape to (total_utterances, num_speakers) for CrossEntropyLoss
        # Each row represents one utterance's similarity to all speaker centroids
        # Shape: (speakers_per_batch * utterances_per_speaker, speakers_per_batch)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                        speakers_per_batch))
        
        # ===== Step 2: Create ground truth labels =====
        # Each utterance should be closest to its speaker's centroid
        # Creates labels like [0,0,0,1,1,1,2,2,2] for 3 speakers with 3 utterances each
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        
        # ===== Step 3: Compute cross entropy loss =====
        # This maximizes similarity between utterances and their true speaker centroids
        # while minimizing similarity to other speaker centroids
        loss = self.loss_fn(sim_matrix, target)
        
        # ===== Step 4: Compute Equal Error Rate (EER) for evaluation =====
        # This is not used for backpropagation, just for tracking performance
        with torch.no_grad():
            # Convert ground truth to one-hot encoding
            # Creates a binary matrix where 1 indicates true speaker
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            
            # Get raw similarity scores
            preds = sim_matrix.detach().cpu().numpy()

            # Calculate EER using ROC curve
            # 1. Compute false positive and true positive rates at different thresholds
            # 2. Find threshold where FAR = FRR (False Accept Rate = False Reject Rate)
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            
            # Find the threshold where FAR = FRR using root finding
            # This is equivalent to finding where TPR = 1-FPR
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer