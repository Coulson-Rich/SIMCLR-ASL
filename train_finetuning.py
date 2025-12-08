import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from pathlib import Path

from src.simclr_model import ISLRDataset, MultimodalEncoder
from src.fine_tune import FineTuneLightning
from src.utils import create_dataset_splits


class FinetuneDataset(ISLRDataset):
    """Dataset variant with label for fine-tuning."""
    
    def __init__(self, metadata, processed_dir='data/processed', 
                 sign_to_label=None, split='train'):
        super().__init__(metadata, processed_dir, transform=None, split=split)
        
        if sign_to_label is None:
            # Create label mapping
            signs = sorted(list(set(item['sign_name'] for item in metadata)))
            sign_to_label = {sign: idx for idx, sign in enumerate(signs)}
        
        self.sign_to_label = sign_to_label
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        sign_name = item['sign_name']
        video_id = item['video_id']
        
        video_dir = self.processed_dir / sign_name / str(video_id)
        
        # Load frames
        import cv2
        frame_files = sorted(video_dir.glob('frame_*.jpg'))
        frames = [cv2.imread(str(f)) for f in frame_files]
        
        # Load pose sequence
        pose_path = video_dir / 'pose_sequence.pkl'
        with open(pose_path, 'rb') as f:
            pose_sequence = pickle.load(f)
        
        # Ensure proper length
        min_len = min(len(frames), len(pose_sequence))
        frames = frames[:min_len]
        pose_sequence = pose_sequence[:min_len]
        
        # Convert to tensors
        import torch
        frames = torch.stack([torch.from_numpy(f).permute(2, 0, 1).float() / 255
                              for f in frames])
        pose = torch.from_numpy(pose_sequence).float()
        label = torch.tensor(self.sign_to_label[sign_name], dtype=torch.long)
        
        return {
            'frames': frames,
            'pose': pose,
            'label': label
        }


def train_finetune(checkpoint_path):
    # Load splits
    with open('data/splits/train.pkl', 'rb') as f:
        train_split = pickle.load(f)
    with open('data/splits/val.pkl', 'rb') as f:
        val_split = pickle.load(f)
    
    # Create label mapping
    signs = sorted(list(set(item['sign_name'] for item in train_split + val_split)))
    sign_to_label = {sign: idx for idx, sign in enumerate(signs)}
    num_classes = len(signs)
    
    # Create datasets
    train_dataset = FinetuneDataset(
        train_split,
        processed_dir='data/processed',
        sign_to_label=sign_to_label,
        split='train'
    )
    
    val_dataset = FinetuneDataset(
        val_split,
        processed_dir='data/processed',
        sign_to_label=sign_to_label,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load pretrained encoder
    encoder = MultimodalEncoder(
        video_feature_dim=2048,
        pose_feature_dim=512,
        fusion_dim=2048
    )
    
    # Load weights from checkpoint
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['state_dict'])
    
    # Create fine-tune model
    model = FineTuneLightning(
        encoder=encoder,
        projection_head=None,
        num_classes=num_classes,
        learning_rate=1e-3,
        freeze_encoder=False  # Fine-tune all layers
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/finetune',
        filename='finetune-{epoch:02d}-{val_accuracy:.3f}',
        monitor='val_accuracy',
        save_top_k=3,
        mode='max'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback],
        default_root_dir='./lightning_logs'
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer


if __name__ == '__main__':
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/simclr-latest.ckpt'
    train_finetune(checkpoint)
