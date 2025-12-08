"""
Main execution script for SimCLR ISLR project.
Run the complete pipeline from preprocessing to evaluation.
"""

import argparse
from pathlib import Path
import torch
import yaml
import pickle

from src.data_preprocessing import VideoPreprocessor
from src.utils import create_dataset_splits
from train_pretraining import train_simclr
from train_finetuning import train_finetune
from src.evaluation import evaluate_model, print_metrics, save_results


def setup():
    """Create necessary directories."""
    dirs = ['data/raw', 'data/processed', 'data/splits', 
            'checkpoints', 'checkpoints/finetune', 'results', 'lightning_logs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def preprocess_data():
    """Step 1: Preprocess videos."""
    print("\n" + "="*60)
    print("STEP 1: PREPROCESSING DATA")
    print("="*60)
    
    preprocessor = VideoPreprocessor(
        output_dir='data/processed',
        target_fps=30,
        target_size=(224, 224)
    )
    
    metadata = preprocessor.process_dataset('data/raw')
    print(f"✓ Preprocessed {len(metadata)} videos")
    
    return metadata


def create_splits():
    """Step 2: Create train/val/test splits."""
    print("\n" + "="*60)
    print("STEP 2: CREATING DATA SPLITS")
    print("="*60)
    
    splits = create_dataset_splits(
        'data/processed/metadata.pkl',
        output_dir='data/splits',
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"✓ Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    return splits


def pretrain():
    """Step 3: Self-supervised pretraining with SimCLR."""
    print("\n" + "="*60)
    print("STEP 3: SIMCLR PRETRAINING")
    print("="*60)
    
    model, trainer = train_simclr()
    
    print("✓ Pretraining completed")
    return model, trainer


def finetune(checkpoint_path):
    """Step 4: Fine-tune for classification."""
    print("\n" + "="*60)
    print("STEP 4: FINE-TUNING FOR CLASSIFICATION")
    print("="*60)
    
    model, trainer = train_finetune(checkpoint_path)
    
    print("✓ Fine-tuning completed")
    return model, trainer


def evaluate(model, checkpoint_path):
    """Step 5: Evaluate on test set."""
    print("\n" + "="*60)
    print("STEP 5: EVALUATION")
    print("="*60)
    
    # Load test data
    with open('data/splits/test.pkl', 'rb') as f:
        test_split = pickle.load(f)
    
    from train_finetuning import FinetuneDataset
    from torch.utils.data import DataLoader
    
    signs = sorted(list(set(item['sign_name'] for item in test_split)))
    sign_to_label = {sign: idx for idx, sign in enumerate(signs)}
    
    test_dataset = FinetuneDataset(
        test_split,
        processed_dir='data/processed',
        sign_to_label=sign_to_label,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    metrics, preds, labels = evaluate_model(model, test_loader, device)
    
    print_metrics(metrics)
    save_results(metrics)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='SimCLR ISLR Pipeline')
    parser.add_argument('--skip-preprocess', action='store_true', 
                       help='Skip preprocessing step')
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip pretraining step')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pretrained checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    setup()
    
    # Execute pipeline
    if not args.skip_preprocess:
        preprocess_data()
        create_splits()
    
    if not args.skip_pretrain:
        model, trainer = pretrain()
        checkpoint = 'checkpoints/simclr-latest.ckpt'
        trainer.save_checkpoint(checkpoint)
    else:
        checkpoint = args.checkpoint or 'checkpoints/simclr-latest.ckpt'
    
    model, trainer = finetune(checkpoint)
    evaluate(model, checkpoint)
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()