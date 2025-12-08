import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from pathlib import Path

from src.simclr_model import (
    ISLRDataset, MultimodalEncoder, SimCLRLightning, ContrastiveTransform
)
from src.utils import create_dataset_splits


def train_simclr():
    # Create dataset splits
    splits = create_dataset_splits(
        'data/processed/metadata.pkl',
        output_dir='data/splits'
    )
    
    # Initialize dataset
    transform = ContrastiveTransform(frame_size=224)
    
    train_dataset = ISLRDataset(
        splits['train'],
        processed_dir='data/processed',
        transform=transform,
        split='train'
    )
    
    val_dataset = ISLRDataset(
        splits['val'],
        processed_dir='data/processed',
        transform=transform,
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
    
    # Initialize model
    encoder = MultimodalEncoder(
        video_feature_dim=2048,
        pose_feature_dim=512,
        fusion_dim=2048
    )
    
    model = SimCLRLightning(
        encoder=encoder,
        projection_dim=128,
        learning_rate=3e-4,
        temperature=0.07
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='simclr-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        default_root_dir='./lightning_logs'
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer


if __name__ == '__main__':
    model, trainer = train_simclr()
