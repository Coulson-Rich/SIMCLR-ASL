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
        self.val_video_dir = self.autsl_root / 'val' / 'val'
        self.test_video_dir = self.autsl_root / 'test' / 'test'
        
        self.train_skel_dir = self.autsl_root / 'train' / 'train_skel'
        self.val_skel_dir = self.autsl_root / 'val' / 'val_skel'
        self.test_skel_dir = self.autsl_root / 'test' / 'test_skel'
        
        self.train_labels_path = self.autsl_root / 'train' / 'train_labels.csv'
        self.val_labels_path = self.autsl_root / 'val' / 'ground_truth.csv'
        self.test_labels_path = self.autsl_root / 'test' / 'ground_truth.csv'
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping()
        
    def _load_class_mapping(self):
        """Load sign class mapping from CSV."""
        df = pd.read_csv(self.class_mapping_path)
        # Returns dict: {ClassId: {'Turkish': ..., 'English': ...}}
        # Handles different CSV column name formats
        tr_col = 'TR' if 'TR' in df.columns else 'Turkish'
        en_col = 'EN' if 'EN' in df.columns else 'English'
        id_col = 'ClassId' if 'ClassId' in df.columns else df.columns[0]
        
        return {
            row[id_col]: {'Turkish': row[tr_col], 'English': row[en_col]}
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

        # Use first two columns: sample_id and class_id
        # Adjust if CSV has explicit column names
        id_col = df.columns[0]      # e.g. 'sample_id'
        class_col = df.columns[1]   # e.g. 'class_id'

        # Keys like 'signer0_sample600'
        return {row[id_col]: row[class_col] for _, row in df.iterrows()}
    
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
        skel_file_path = Path(skel_file_path)
        
        if skel_file_path.suffix == '.npy':
            pose_sequence = np.load(skel_file_path)
        elif skel_file_path.suffix == '.pkl':
            with open(skel_file_path, 'rb') as f:
                pose_sequence = pickle.load(f)
        else:
            raise ValueError(f"Unsupported skeleton file format: {skel_file_path.suffix}")
        
        # Handle shape: if [T, num_keypoints, 4], flatten to [T, num_keypoints*4]
        if pose_sequence.ndim == 3:
            T, num_kpts, num_vals = pose_sequence.shape
            pose_sequence = pose_sequence.reshape(T, num_kpts * num_vals)
        
        # Expected shape after this: (num_frames, pose_dim)
        # e.g., (90, 132) for 33 keypoints × 4 values
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

        labels = self._load_labels(split)
        metadata = []
        video_files = sorted(video_dir.glob('*.mp4'))

        for video_path in tqdm(video_files, desc=f"Loading {split} metadata"):
            stem = video_path.stem              # e.g. 'signer10_sample151_color' or 'signer10_sample151_depth'
            
            # Extract base_id by removing '_color' or '_depth' suffix
            if stem.endswith('_color'):
                base_id = stem[:-len('_color')]  # 'signer10_sample151'
            elif stem.endswith('_depth'):
                base_id = stem[:-len('_depth')]  # 'signer10_sample151'
            else:
                base_id = stem

            # USE ONLY COLOR LANDMARKS
            # Skeleton filenames are: base_id + '_color_landmarks.npy'
            skel_path = skel_dir / f"{base_id}_color_landmarks.npy"
            
            if not skel_path.exists():
                continue

            # Labels are keyed by base_id
            class_id = labels.get(base_id, None)
            if class_id is None:
                continue

            metadata.append({
                'video_id': base_id,
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
