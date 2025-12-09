import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle

class AUTSLDataLoader:
    """Load AUTSL videos and pre-extracted pose keypoints."""
    
    def __init__(self, autsl_root='/home/ccoulson/groups/grp_asl_classification/nobackup/archive/AUTSL'):
        self.autsl_root = Path(autsl_root)
        
        # Define paths
        self.class_mapping_path = self.autsl_root / 'class_ids' / 'SignList_ClassId_TR_EN.csv'
        
        self.train_video_dir = self.autsl_root / 'train' / 'train'
        self.val_video_dir = self.autsl_root / 'validate' / 'validate'
        self.test_video_dir = self.autsl_root / 'test' / 'test'
        
        self.train_skel_dir = self.autsl_root / 'train' / 'train_skel'
        self.val_skel_dir = self.autsl_root / 'validate' / 'validate_skel'
        self.test_skel_dir = self.autsl_root / 'test' / 'test_skel'
        
        self.train_labels_path = self.autsl_root / 'train' / 'train_labels.csv'
        self.val_labels_path = self.autsl_root / 'validate' / 'ground_truth.csv'
        self.test_labels_path = self.autsl_root / 'test' / 'ground_truth.csv'
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping()
        
    def _load_class_mapping(self):
        """Load sign class mapping from CSV."""
        df = pd.read_csv(self.class_mapping_path)
        # Returns dict: {ClassId: {'Turkish': ..., 'English': ...}}
        return {
            row['ClassId']: {'Turkish': row['Turkish'], 'English': row['English']}
            for _, row in df.iterrows()
        }
    
    def _load_labels(self, split='train'):
        """Load labels for a given split."""
        if split == 'train':
            df = pd.read_csv(self.train_labels_path)
        elif split == 'val':
            df = pd.read_csv(self.val_labels_path)
        elif split == 'test':
            df = pd.read_csv(self.test_labels_path)
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Assuming CSV has columns: video_id, class_id (or similar)
        # Adjust column names based on actual AUTSL format
        return {row[0]: row[1] for row in df.values}  # {video_id: class_id}
    
    def load_video_frames(self, video_path, target_size=(224, 224), max_frames=90):
        """Load and resize video frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def load_pose_keypoints(self, skel_file_path):
        """Load pre-extracted pose keypoints from AUTSL skeleton files."""
        # AUTSL skeleton files are typically .npy or .pkl format
        # Adjust based on actual AUTSL skeleton format
        
        if skel_file_path.suffix == '.npy':
            pose_sequence = np.load(skel_file_path)
        elif skel_file_path.suffix == '.pkl':
            with open(skel_file_path, 'rb') as f:
                pose_sequence = pickle.load(f)
        else:
            raise ValueError(f"Unsupported skeleton file format: {skel_file_path.suffix}")
        
        # Expected shape: (num_frames, num_keypoints * coordinates)
        # AUTSL may have different format - adjust accordingly
        return pose_sequence
    
    def create_dataset_metadata(self, split='train'):
        """Create metadata for dataset split."""
        if split == 'train':
            video_dir = self.train_video_dir
            skel_dir = self.train_skel_dir
        elif split == 'val':
            video_dir = self.val_video_dir
            skel_dir = self.val_skel_dir
        elif split == 'test':
            video_dir = self.test_video_dir
            skel_dir = self.test_skel_dir
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Load labels
        labels = self._load_labels(split)
        
        metadata = []
        video_files = sorted(video_dir.glob('*.mp4'))
        
        for video_path in tqdm(video_files, desc=f"Loading {split} metadata"):
            video_id = video_path.stem
            
            # Find corresponding skeleton file
            skel_path = skel_dir / f"{video_id}.npy"  # Adjust extension if needed
            if not skel_path.exists():
                skel_path = skel_dir / f"{video_id}.pkl"
            
            if not skel_path.exists():
                print(f"Warning: No skeleton file found for {video_id}")
                continue
            
            # Get class label
            class_id = labels.get(video_id, None)
            if class_id is None:
                print(f"Warning: No label found for {video_id}")
                continue
            
            metadata.append({
                'video_id': video_id,
                'video_path': str(video_path),
                'skel_path': str(skel_path),
                'class_id': class_id,
                'split': split
            })
        
        print(f"✓ Loaded {len(metadata)} samples for {split} split")
        return metadata
    
    def save_metadata(self, output_dir='data'):
        """Create and save metadata for all splits."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'train': self.create_dataset_metadata('train'),
            'val': self.create_dataset_metadata('val'),
            'test': self.create_dataset_metadata('test')
        }
        
        # Save metadata
        for split, data in metadata.items():
            output_path = output_dir / f'{split}_metadata.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved {split} metadata to {output_path}")
        
        # Save class mapping
        class_mapping_path = output_dir / 'class_mapping.pkl'
        with open(class_mapping_path, 'wb') as f:
            pickle.dump(self.class_mapping, f)
        print(f"✓ Saved class mapping to {class_mapping_path}")
        
        return metadata


# Run data loading
if __name__ == '__main__':
    loader = AUTSLDataLoader()
    metadata = loader.save_metadata(output_dir='data')
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Train samples: {len(metadata['train'])}")
    print(f"Val samples: {len(metadata['val'])}")
    print(f"Test samples: {len(metadata['test'])}")
    print(f"Total classes: {len(loader.class_mapping)}")
    print("="*60)