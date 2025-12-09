import pickle
from src.simclr_model import AUTSLDataset
from src.augmentation import ContrastiveTransform

with open('data/train_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)

transform = ContrastiveTransform(frame_size=224, max_frames=16)
ds = AUTSLDataset(meta, transform=transform)
sample = ds[0]
print('frames1:', sample['frames1'].shape)
print('pose1:', sample['pose1'].shape)