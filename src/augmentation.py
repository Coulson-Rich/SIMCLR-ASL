import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
import cv2
from PIL import Image

class SignLanguageAugmentations:
    """Sign language-specific augmentations for contrastive learning."""
    
    def __init__(self, frame_size=224):
        self.frame_size = frame_size
    
    def temporal_crop(self, frames, min_ratio=0.7, max_ratio=0.95):
        """Temporally crop video while preserving sign semantics."""
        num_frames = len(frames)
        target_length = random.randint(
            int(num_frames * min_ratio),
            int(num_frames * max_ratio)
        )
        
        max_start = num_frames - target_length
        start_idx = random.randint(0, max(0, max_start))
        
        return frames[start_idx:start_idx + target_length]
    
    def temporal_jitter(self, frames, max_jitter=3):
        """Randomly sample frames with small jitter."""
        num_frames = len(frames)
        sample_rate = max(1, num_frames // 16)
        
        indices = []
        for i in range(0, num_frames, sample_rate):
            jitter = random.randint(-max_jitter, max_jitter)
            idx = max(0, min(num_frames - 1, i + jitter))
            indices.append(idx)
        
        return [frames[i] for i in indices]
    
    def spatial_transforms(self, frame):
        """Apply spatial augmentations to individual frame."""
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if random.random() > 0.5:
            frame = TF.center_crop(frame, (int(self.frame_size * 0.9), 
                                           int(self.frame_size * 0.9)))
            frame = TF.resize(frame, (self.frame_size, self.frame_size))
        
        if random.random() > 0.5:
            frame = TF.hflip(frame)
        
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                              saturation=0.1, hue=0.05)
        frame = color_jitter(frame)
        
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            frame = TF.rotate(frame, angle, expand=False)
        
        if random.random() > 0.7:
            frame = TF.gaussian_blur(frame, kernel_size=5)
        
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    
    def pose_augmentation(self, pose_sequence, noise_std=0.01):
        """Augment pose keypoint sequence."""
        augmented = pose_sequence.copy()
        
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented = augmented + noise
        
        if random.random() > 0.7:
            window = 3
            for i in range(window, len(augmented) - window):
                augmented[i] = np.mean(
                    augmented[i-window:i+window+1], axis=0
                )
        
        return augmented
    
    def get_augmentation_pair(self, video_frames, pose_sequence):
        """Generate two augmented views."""
        frames1 = self.temporal_crop(video_frames)
        frames1 = self.temporal_jitter(frames1)
        frames1 = [self.spatial_transforms(f) for f in frames1]
        pose1 = self.pose_augmentation(pose_sequence[:len(frames1)])
        
        frames2 = self.temporal_crop(video_frames)
        frames2 = self.temporal_jitter(frames2)
        frames2 = [self.spatial_transforms(f) for f in frames2]
        pose2 = self.pose_augmentation(pose_sequence[:len(frames2)])
        
        return frames1, frames2, pose1, pose2


class ContrastiveTransform:
    """Wrapper for generating contrastive pairs."""
    
    def __init__(self, frame_size=224, max_frames=16):
        self.augment = SignLanguageAugmentations(frame_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.max_frames = max_frames

    def _pad_or_trim(self, frames, target_len):
        """Pad with last frame or trim to get exactly target_len frames."""
        n = len(frames)
        if n == 0:
            raise ValueError("Received empty frame sequence in augmentation.")
        if n > target_len:
            return frames[:target_len]
        if n < target_len:
            last = frames[-1]
            frames = frames + [last] * (target_len - n)
        return frames
    
    def __call__(self, video_frames, pose_sequence):
        """Generate augmented pair for contrastive learning."""
        frames1, frames2, pose1, pose2 = \
            self.augment.get_augmentation_pair(video_frames, pose_sequence)

        # Enforce fixed length
        frames1 = self._pad_or_trim(frames1, self.max_frames)
        frames2 = self._pad_or_trim(frames2, self.max_frames)
        
        # Trim pose sequences to match
        pose1 = pose1[:self.max_frames]
        pose2 = pose2[:self.max_frames]
        
        # Pad pose if needed
        if len(pose1) < self.max_frames:
            pad_len = self.max_frames - len(pose1)
            pose1 = np.vstack([pose1, np.tile(pose1[-1], (pad_len, 1))])
        if len(pose2) < self.max_frames:
            pad_len = self.max_frames - len(pose2)
            pose2 = np.vstack([pose2, np.tile(pose2[-1], (pad_len, 1))])

        # Convert frames to tensors with normalization
        imgs1 = torch.stack([self.normalize(self.to_tensor(f)) for f in frames1])
        imgs2 = torch.stack([self.normalize(self.to_tensor(f)) for f in frames2])

        # Convert poses to tensors
        pose1 = torch.from_numpy(pose1).float()
        pose2 = torch.from_numpy(pose2).float()
        
        return imgs1, imgs2, pose1, pose2
