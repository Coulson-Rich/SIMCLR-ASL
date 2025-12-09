import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle
import pandas as pd

from src.simclr_model import AUTSLDataset, MultimodalEncoder
from src.fine_tune import FineTuneLightning


class FineTuneAUTSLDataset(AUTSLDataset):
    """Dataset variant with labels for fine-tuning."""
    
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        # Convert class_id to tensor label
        result['label'] = torch.tensor(result['class_id'], dtype=torch.long)
        return result


def train_finetune(checkpoint_path):
    # Load metadata
    with open('data/train_metadata.pkl', 'rb') as f:
        train_metadata = pickle.load(f)
    with open('data/val_metadata.pkl', 'rb') as f:
        val_metadata = pickle.load(f)
    with open('data/class_mapping.pkl', 'rb') as f:
        class_mapping = pickle.load(f)
    
    num_classes = len(class_mapping)
    
    # Create datasets
    train_dataset = FineTuneAUTSLDataset(train_metadata, transform=None, split='train')
    val_dataset = FineTuneAUTSLDataset(val_metadata, transform=None, split='val')
    
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
        fusion_dim=2048,
        pose_input_dim=132
    )
    
    # Load weights from checkpoint
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    
    # Create fine-tune model
    model = FineTuneLightning(
        encoder=encoder,
        num_classes=num_classes,
        learning_rate=1e-3,
        freeze_encoder=False
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
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
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