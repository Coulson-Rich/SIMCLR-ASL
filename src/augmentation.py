import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random

class SignLanguageAugmentations:
    """Sign language-specific augmentations for contrastive learning."""
    
    def __init__(self, frame_size=224):
        self.frame_size = frame_size
    
    def temporal_crop(self, frames, min_ratio=0.7, max_ratio=0.95):
        """
        Temporally crop video while preserving sign semantics.
        Avoid cropping too much from beginning/end.
        """
        num_frames = len(frames)
        target_length = random.randint(
            int(num_frames * min_ratio),
            int(num_frames * max_ratio)
        )
        
        # Leave margin at start and end
        max_start = num_frames - target_length
        start_idx = random.randint(0, max(0, max_start))
        
        return frames[start_idx:start_idx + target_length]
    
    def temporal_jitter(self, frames, max_jitter=3):
        """Randomly sample frames with small jitter."""
        num_frames = len(frames)
        sample_rate = max(1, num_frames // 16)  # Target ~16 frames
        
        indices = []
        for i in range(0, num_frames, sample_rate):
            jitter = random.randint(-max_jitter, max_jitter)
            idx = max(0, min(num_frames - 1, i + jitter))
            indices.append(idx)
        
        return [frames[i] for i in indices]
    
    def spatial_transforms(self, frame):
        """Apply spatial augmentations to individual frame."""
        # Convert numpy array to PIL Image for transforms
        from PIL import Image
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Random crop
        if random.random() > 0.5:
            frame = TF.center_crop(frame, (int(self.frame_size * 0.9), 
                                           int(self.frame_size * 0.9)))
            frame = TF.resize(frame, (self.frame_size, self.frame_size))
        
        # Random horizontal flip (mirror for sign language)
        if random.random() > 0.5:
            frame = TF.hflip(frame)
        
        # Color jitter
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                              saturation=0.1, hue=0.05)
        frame = color_jitter(frame)
        
        # Random rotation (small angle to preserve sign)
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            frame = TF.rotate(frame, angle, expand=False)
        
        # Gaussian blur
        if random.random() > 0.7:
            frame = TF.gaussian_blur(frame, kernel_size=5)
        
        # Convert back to numpy array
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    
    def pose_augmentation(self, pose_sequence, noise_std=0.01):
        """
        Augment pose keypoint sequence.
        Add realistic noise to handle pose estimation uncertainty.
        """
        augmented = pose_sequence.copy()
        
        # Add Gaussian noise to coordinates
        # Adjust based on actual AUTSL pose format
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented = augmented + noise
        
        # Temporal smoothing (running average to handle jitter)
        if random.random() > 0.7:
            window = 3
            for i in range(window, len(augmented) - window):
                augmented[i] = np.mean(
                    augmented[i-window:i+window+1], axis=0
                )
        
        return augmented
    
    def get_augmentation_pair(self, video_frames, pose_sequence):
        """
        Generate two augmented views of the same video for contrastive learning.
        Returns: (frames1, frames2, pose1, pose2)
        """
        # Augmentation 1
        frames1 = self.temporal_crop(video_frames)
        frames1 = self.temporal_jitter(frames1)
        frames1 = [self.spatial_transforms(f) for f in frames1]
        pose1 = self.pose_augmentation(pose_sequence[:len(frames1)])
        
        # Augmentation 2 (different strategy)
        frames2 = self.temporal_crop(video_frames)
        frames2 = self.temporal_jitter(frames2)
        frames2 = [self.spatial_transforms(f) for f in frames2]
        pose2 = self.pose_augmentation(pose_sequence[:len(frames2)])
        
        return frames1, frames2, pose1, pose2


class ContrastiveTransform:
    """Wrapper for generating contrastive pairs."""
    
    def __init__(self, frame_size=224):
        self.augment = SignLanguageAugmentations(frame_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, video_frames, pose_sequence):
        """
        Generate augmented pair for contrastive learning.
        Returns: (image1, image2, pose1, pose2)
        """
        frames1, frames2, pose1, pose2 = \
            self.augment.get_augmentation_pair(video_frames, pose_sequence)
        
        # Convert frames to tensors with normalization
        imgs1 = torch.stack([self.normalize(self.to_tensor(f)) for f in frames1])
        imgs2 = torch.stack([self.normalize(self.to_tensor(f)) for f in frames2])
        
        # Convert poses to tensors
        pose1 = torch.from_numpy(pose1).float()
        pose2 = torch.from_numpy(pose2).float()
        
        return imgs1, imgs2, pose1, pose2