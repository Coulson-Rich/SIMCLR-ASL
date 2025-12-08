import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle

class VideoPreprocessor:
    """Extract frames and pose keypoints from sign language videos."""
    
    def __init__(self, output_dir='data/processed', target_fps=30, target_size=(224, 224)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_fps = target_fps
        self.target_size = target_size
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_frames(self, video_path, sign_name, max_frames=None):
        """Extract frames from video and save as images."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling rate
        sample_rate = max(1, int(fps / self.target_fps))
        
        frames = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # Resize frame
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def extract_pose_keypoints(self, frame):
        """Extract pose landmarks from a single frame using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract 33 landmarks (x, y, z, visibility)
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            return np.array(landmarks, dtype=np.float32)  # Shape: (132,)
        else:
            # Return zeros if no pose detected
            return np.zeros((132,), dtype=np.float32)
    
    def process_video(self, video_path, sign_name, video_id):
        """Process a single video: extract frames and pose keypoints."""
        video_output_dir = self.output_dir / sign_name / str(video_id)
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        frames = self.extract_frames(video_path, sign_name, max_frames=90)
        
        if len(frames) == 0:
            print(f"Warning: No frames extracted from {video_path}")
            return None
        
        # Extract pose keypoints for all frames
        pose_sequence = []
        frame_files = []
        
        for frame_idx, frame in enumerate(frames):
            # Save frame
            frame_path = video_output_dir / f'frame_{frame_idx:04d}.jpg'
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(str(frame_path))
            
            # Extract pose
            pose_keypoints = self.extract_pose_keypoints(frame)
            pose_sequence.append(pose_keypoints)
        
        # Save pose sequence
        pose_path = video_output_dir / 'pose_sequence.pkl'
        with open(pose_path, 'wb') as f:
            pickle.dump(np.array(pose_sequence), f)
        
        # Save metadata
        metadata = {
            'video_id': video_id,
            'sign_name': sign_name,
            'num_frames': len(frames),
            'frame_files': frame_files,
            'pose_path': str(pose_path)
        }
        
        return metadata
    
    def process_dataset(self, raw_data_dir):
        """Process entire dataset."""
        raw_data_dir = Path(raw_data_dir)
        all_metadata = []
        
        # Assuming directory structure: raw_data_dir/sign_name/video.mp4
        for sign_dir in tqdm(raw_data_dir.iterdir(), desc="Processing signs"):
            if not sign_dir.is_dir():
                continue
            
            sign_name = sign_dir.name
            video_id = 0
            
            for video_file in sign_dir.glob('*.mp4'):
                metadata = self.process_video(video_file, sign_name, video_id)
                if metadata:
                    all_metadata.append(metadata)
                video_id += 1
        
        # Save all metadata
        metadata_path = self.output_dir / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)
        
        print(f"Processed {len(all_metadata)} videos")
        return all_metadata


# Run preprocessing
if __name__ == '__main__':
    preprocessor = VideoPreprocessor(
        output_dir='data/processed',
        target_fps=30,
        target_size=(224, 224)
    )
    
    metadata = preprocessor.process_dataset('data/raw')
    print(f"Total videos processed: {len(metadata)}")