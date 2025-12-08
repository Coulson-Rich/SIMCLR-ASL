import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import torch
from pathlib import Path


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            frames = batch['frames'].to(device)
            pose = batch['pose'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(frames, pose)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    
    return metrics, all_preds, all_labels


def print_metrics(metrics):
    """Print evaluation metrics."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    print("="*50 + "\n")


def save_results(metrics, output_dir='results'):
    """Save evaluation results to file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(Path(output_dir) / 'metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")


def ablation_study(model, test_loader, device='cuda'):
    """
    Ablation study: evaluate model with different input modalities.
    - Video only
    - Pose only
    - Both (multimodal)
    """
    # This is a placeholder for modality ablation
    # You would need to modify the model to support this
    pass
