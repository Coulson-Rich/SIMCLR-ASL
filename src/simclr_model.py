import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import pickle
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR."""
    
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class VideoEncoder(nn.Module):
    """Temporal encoder for video frames using 3D CNN."""
    
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import resnet50
        
        # No internet-required weights; random init
        self.cnn = resnet50(weights=None)
        
        # Remove classification head
        self.encoder = nn.Sequential(*list(self.cnn.children())[:-1])
        self.feature_dim = 2048
    
    def forward(self, x):
        """
        x: (batch, time, channels, height, width)
        Returns: (batch, feature_dim)
        """
        batch_size, time_steps, c, h, w = x.shape
        
        # Process each frame
        x = x.view(batch_size * time_steps, c, h, w)
        features = self.encoder(x)  # (batch*time, 2048, 1, 1)
        features = features.view(batch_size * time_steps, -1)
        
        # Temporal pooling (mean across time)
        features = features.view(batch_size, time_steps, -1)
        features = features.mean(dim=1)  # (batch, 2048)
        
        return features


class PoseEncoder(nn.Module):
    """Encoder for pose keypoint sequences."""
    
    def __init__(self, input_dim=2172, hidden_dim=256, output_dim=512):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.feature_dim = output_dim
    
    def forward(self, x):
        """
        x: (batch, time_steps, pose_dim)
        Returns: (batch, output_dim)
        """
        # Handle edge cases
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"PoseEncoder expects 3D input [batch, time, pose_dim], got shape {x.shape}")
        
        _, (h_n, _) = self.lstm(x)
        features = h_n[-1]
        features = self.fc(features)
        return features


class MultimodalEncoder(nn.Module):
    """Combines video and pose encoders."""
    
    def __init__(self, video_feature_dim=2048, pose_feature_dim=512, 
                 fusion_dim=2048, pose_input_dim=2172):
        super().__init__()
        self.video_encoder = VideoEncoder(pretrained=False)
        self.pose_encoder = PoseEncoder(input_dim=pose_input_dim, output_dim=pose_feature_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(video_feature_dim + pose_feature_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim)
        )
        self.feature_dim = fusion_dim
    
    def forward(self, video_frames, pose_sequence):
        """
        video_frames: (batch, time, c, h, w)
        pose_sequence: (batch, time, pose_dim)
        """
        video_features = self.video_encoder(video_frames)
        pose_features = self.pose_encoder(pose_sequence)
        
        # Concatenate and fuse
        combined = torch.cat([video_features, pose_features], dim=1)
        fused_features = self.fusion(combined)
        
        return fused_features


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss."""
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, z_i, z_j):
        """
        z_i, z_j: (batch_size, feature_dim)
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate [z_i, z_j] -> shape [2*batch_size, feature_dim]
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix: [2*batch_size, 2*batch_size]
        similarity_matrix = torch.mm(z, z.T) / self.temperature
        
        # Create mask for positive pairs
        batch_size_2x = 2 * batch_size
        mask = torch.eye(batch_size_2x, dtype=torch.bool, device=z.device)
        
        # Get positive sample indices
        positives = torch.cat([
            torch.arange(batch_size, batch_size_2x),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        # Remove diagonal (self-similarity)
        similarity_matrix.masked_fill_(mask, -1e9)
        
        # Labels for cross entropy: index of positive sample
        labels = positives
        
        # Compute loss
        loss = self.criterion(similarity_matrix, labels) / batch_size_2x
        
        return loss


class AUTSLDataset(Dataset):
    """PyTorch Dataset for AUTSL."""
    
    def __init__(self, metadata, transform=None, split='train'):
        self.metadata = metadata
        self.transform = transform
        self.split = split
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load video frames
        cap = cv2.VideoCapture(item['video_path'])
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        
        # Load pre-extracted pose
        skel_path = Path(item['skel_path'])
        if skel_path.suffix == '.npy':
            pose_sequence = np.load(skel_path)
        else:
            with open(skel_path, 'rb') as f:
                pose_sequence = pickle.load(f)
        
        # Flatten if 3D: [T, num_kpts, 4] -> [T, num_kpts*4]
        if pose_sequence.ndim == 3:
            T, num_kpts, num_vals = pose_sequence.shape
            pose_sequence = pose_sequence.reshape(T, num_kpts * num_vals)
        
        # **FIX NaNs with interpolation**
        # Convert to pandas for easier interpolation, then back to numpy
        pose_sequence = pd.DataFrame(pose_sequence).interpolate(
            method='linear', axis=0, limit_direction='both'
        ).fillna(0).values
        
        # Ensure matching lengths
        min_len = min(len(frames), len(pose_sequence))
        frames = frames[:min_len]
        pose_sequence = pose_sequence[:min_len]
        
        # Apply augmentations if transform provided
        if self.transform:
            frames1, frames2, pose1, pose2 = \
                self.transform(frames, pose_sequence)
            return {
                'frames1': frames1,
                'frames2': frames2,
                'pose1': pose1,
                'pose2': pose2,
                'class_id': item['class_id']
            }
        else:
            # For evaluation, return single view
            frames_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 
                for f in frames
            ])
            pose_tensor = torch.from_numpy(pose_sequence).float()
            
            return {
                'frames': frames_tensor,
                'pose': pose_tensor,
                'class_id': item['class_id']
            }


class SimCLRLightning(pl.LightningModule):
    """PyTorch Lightning module for SimCLR training."""
    
    def __init__(self, encoder, projection_dim=128, learning_rate=1e-4, 
                 temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.projection_head = ProjectionHead(
            input_dim=encoder.feature_dim,
            output_dim=projection_dim
        )
        self.loss_fn = NTXentLoss(temperature=temperature)
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['encoder'])
    
    def forward(self, video, pose):
        h = self.encoder(video, pose)
        z = self.projection_head(h)
        return h, z
    
    def training_step(self, batch, batch_idx):
        h1, z1 = self(batch['frames1'], batch['pose1'])
        h2, z2 = self(batch['frames2'], batch['pose2'])
        
        loss = self.loss_fn(z1, z2)
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        h1, z1 = self(batch['frames1'], batch['pose1'])
        h2, z2 = self(batch['frames2'], batch['pose2'])
        
        loss = self.loss_fn(z1, z2)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-6
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
