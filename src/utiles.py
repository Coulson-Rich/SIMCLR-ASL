import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_dataset_splits(metadata_path, output_dir='data/splits', 
                         train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test splits stratified by sign class."""
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by sign class
    sign_classes = {}
    for item in metadata:
        sign = item['sign_name']
        if sign not in sign_classes:
            sign_classes[sign] = []
        sign_classes[sign].append(item)
    
    train_split = []
    val_split = []
    test_split = []
    
    # Stratified split by sign class
    for sign, items in sign_classes.items():
        indices = np.arange(len(items))
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=1-train_ratio-val_ratio, 
            random_state=seed
        )
        
        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx, 
            test_size=val_ratio/(train_ratio+val_ratio),
            random_state=seed
        )
        
        train_split.extend([items[i] for i in train_idx])
        val_split.extend([items[i] for i in val_idx])
        test_split.extend([items[i] for i in test_idx])
    
    # Save splits
    splits = {
        'train': train_split,
        'val': val_split,
        'test': test_split
    }
    
    for split_name, split_data in splits.items():
        path = output_dir / f'{split_name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(split_data, f)
    
    print(f"Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")
    return splits