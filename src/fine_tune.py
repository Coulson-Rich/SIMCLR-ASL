import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
import numpy as np

class ClassificationHead(nn.Module):
    """Classification head for fine-tuning."""
    
    def __init__(self, input_dim=2048, num_classes=226, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)


class FineTuneLightning(pl.LightningModule):
    """Fine-tuning module on top of pretrained encoder."""
    
    def __init__(self, encoder, num_classes, learning_rate=1e-3, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.classifier = ClassificationHead(
            input_dim=encoder.feature_dim,
            num_classes=num_classes
        )
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.num_classes = num_classes
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.save_hyperparameters(ignore=['encoder'])
        
        # For metrics tracking
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
    
    def forward(self, video, pose):
        h = self.encoder(video, pose)
        logits = self.classifier(h)
        return logits
    
    def training_step(self, batch, batch_idx):
        frames = batch['frames']
        pose = batch['pose']
        labels = batch['label']
        
        logits = self(frames, pose)
        loss = F.cross_entropy(logits, labels)
        
        preds = logits.argmax(dim=1)
        self.train_preds.extend(preds.cpu().numpy())
        self.train_labels.extend(labels.cpu().numpy())
        
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        frames = batch['frames']
        pose = batch['pose']
        labels = batch['label']
        
        logits = self(frames, pose)
        loss = F.cross_entropy(logits, labels)
        
        preds = logits.argmax(dim=1)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_labels.extend(labels.cpu().numpy())
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        if len(self.train_labels) > 0:
            acc = accuracy_score(self.train_labels, self.train_preds)
            self.log('train_accuracy', acc, on_epoch=True, prog_bar=True)
            self.train_preds = []
            self.train_labels = []
    
    def on_validation_epoch_end(self):
        if len(self.val_labels) > 0:
            acc = accuracy_score(self.val_labels, self.val_preds)
            self.log('val_accuracy', acc, on_epoch=True, prog_bar=True)
            self.val_preds = []
            self.val_labels = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }