import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pickle

from src.simclr_model import AUTSLDataset, MultimodalEncoder, SimCLRLightning
from src.augmentations import ContrastiveTransform


def train_simclr():
    # Load metadata
    with open('data/train_metadata.pkl', 'rb') as f:
        train_metadata = pickle.load(f)
    with open('data/val_metadata.pkl', 'rb') as f:
        val_metadata = pickle.load(f)
    
    # Initialize dataset
    transform = ContrastiveTransform(frame_size=224)
    
    train_dataset = AUTSLDataset(train_metadata, transform=transform, split='train')
    val_dataset = AUTSLDataset(val_metadata, transform=transform, split='val')
    
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
        fusion_dim=2048,
        pose_input_dim=132  # Adjust based on AUTSL pose format
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
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        default_root_dir='./lightning_logs'
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer


if __name__ == '__main__':
    model, trainer = train_simclr()