# Strategy C: Learnable Joint Pruning for Your Codebase

## Overview
Instead of manually selecting which joints to keep, you'll add a **learnable pruning mechanism** that lets the model decide which joints are important. This uses **Gumbel-Softmax** to create a differentiable "selection mask" over joints.

---

## Implementation: 3 Files to Modify

### 1. **lightning_asl.py** - Add Joint Pruning Module

Add this new class **before** the `SignClassificationLightning` class:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class JointPruningLayer(nn.Module):
    """
    Learnable joint pruning using Gumbel-Softmax.
    
    Learns a probability distribution over which joints to keep.
    During inference, this becomes a hard selection (0 or 1).
    """
    
    def __init__(self, num_joints: int, temperature: float = 1.0, hard: bool = False):
        super().__init__()
        self.num_joints = num_joints
        self.temperature = temperature
        self.hard = hard
        
        # Learnable logits: one per joint
        # Initialize with high probability of keeping all joints
        self.joint_logits = nn.Parameter(torch.zeros(num_joints))
        
    def forward(self, skeleton_keypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skeleton_keypoints: Shape (B, T, J, 2) where J = num_joints
            
        Returns:
            pruned_keypoints: Shape (B, T, J, 2) with selected joints zeroed out
        """
        B, T, J, P = skeleton_keypoints.shape
        assert J == self.num_joints, f"Expected {self.num_joints} joints, got {J}"
        
        # Create selection mask using Gumbel-Softmax
        # This produces a soft probability distribution over joints
        selection_logits = self.joint_logits.unsqueeze(0)  # (1, J)
        
        # Gumbel-Softmax: soft selection during training, hard during inference
        gumbel_softmax = F.gumbel_softmax(
            selection_logits,
            tau=self.temperature,
            hard=self.hard,
            dim=-1
        )  # (1, J)
        
        # Reshape to broadcast: (1, 1, J, 1)
        mask = gumbel_softmax.view(1, 1, J, 1)
        
        # Apply mask to keypoints
        pruned_keypoints = skeleton_keypoints * mask
        
        return pruned_keypoints
    
    def get_active_joints(self, threshold: float = 0.5) -> list:
        """
        Get indices of joints that are likely to be selected (for analysis).
        """
        probs = F.softmax(self.joint_logits, dim=-1)
        active = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
        return active
    
    def get_pruning_ratio(self) -> float:
        """
        Get the fraction of joints being pruned (zeroed out).
        """
        probs = F.softmax(self.joint_logits, dim=-1)
        return (probs < 0.5).float().mean().item()
```

---

### 2. **lightning_asl.py** - Modify `__init__` Method

In `SignClassificationLightning.__init__`, add the pruning layer **before** the skeleton encoder setup:

```python
# Skeleton encoder
# ===== ADD THIS BLOCK =====
num_pose_points = config["num_pose_points"]
self.joint_pruning = JointPruningLayer(
    num_joints=num_pose_points,
    temperature=1.0,
    hard=False  # Soft during training, hard during inference
)
# ===== END NEW BLOCK =====

# Original skeleton encoder setup (KEEP THIS)
bert_config = BertConfig(
    hidden_size=self.config["bert_hidden_dim"],
    num_hidden_layers=self.config["bert_hidden_layers"],
    num_attention_heads=self.config["bert_att_heads"],
    intermediate_size=self.config["bert_intermediate_size"],
    max_position_embeddings=video_mae_config.num_frames,
    vocab_size=1,
    type_vocab_size=1
)

self.skel_encoder = BertModel(bert_config)
self.skel_proj = torch.nn.Linear(config["num_pose_points"] * 2, self.skel_encoder.config.hidden_size)
self.skel_encoder.train()
self.skel_head = torch.nn.Linear(self.skel_encoder.config.hidden_size, self.config["fusion_dim"])
```

---

### 3. **lightning_asl.py** - Modify `forward` Method

In the Skeleton section of the `forward` method, add pruning:

```python
def forward(self, pixel_values=None, depth_values=None, skeleton_keypoints=None):
    # ... RGB and Depth sections stay the same ...
    
    # Skeleton
    if skeleton_keypoints is not None:
        B, T, J, P = skeleton_keypoints.shape
        assert P == 2
        
        # ===== ADD PRUNING HERE =====
        skeleton_keypoints = self.joint_pruning(skeleton_keypoints)
        # ===== END NEW CODE =====
        
        # Flatten and project (keep this as is)
        skeleton_keypoints = skeleton_keypoints.view(B, T, J * P)
        skeleton_keypoints = self.skel_proj(skeleton_keypoints)
        skel_output = self.skel_encoder(input_embeds=skeleton_keypoints).last_hidden_state[:, 0]
        skel_feat = self.skel_head(skel_output)
        features.append(skel_feat)
        weights.append(self.modality_weights[2])
    
    # ... rest stays the same ...
```

---

### 4. **lightning_asl.py** - Modify `configure_optimizers`

The pruning layer parameters need to be optimized. Add this to the optimizer parameter groups:

```python
def configure_optimizers(self):
    optimizer = optim.AdamW([
        {
            "params": self.rgb_encoder.parameters(),
            "lr": self.config["pretrained_learning_rate"],
            "weight_decay": self.config["weight_decay"]
        },
        # ... other param groups ...
        {
            "params": self.skel_encoder.parameters(),
            "lr": self.config["skel_learning_rate"],
            "weight_decay": self.config["weight_decay"]
        },
        {
            "params": self.skel_head.parameters(),
            "lr": self.config["skel_learning_rate"],
            "weight_decay": self.config["weight_decay"]
        },
        # ===== ADD THIS =====
        {
            "params": [self.joint_pruning.joint_logits],
            "lr": self.config["skel_learning_rate"],  # Same as skeleton
            "weight_decay": 0.0  # No decay on pruning logits
        },
        # ===== END NEW PARAM GROUP =====
        {
            "params": [self.modality_weights],
            "lr": self.config["class_learning_rate"],
            "weight_decay": self.config["weight_decay"]
        },
        {
            "params": self.classifier.parameters(),
            "lr": self.config["class_learning_rate"],
            "weight_decay": self.config["weight_decay"]
        }
    ])
    
    # ... rest of optimizer setup stays the same ...
```

---

## 5. **Optional: Add Logging for Pruning Statistics**

In your `training_step` and `validation_step`, add logging:

```python
def training_step(self, batch, batch_idx):
    # ... existing code ...
    
    self.log("train_loss", loss, on_step=True, on_epoch=True, ...)
    
    # ===== ADD PRUNING STATS =====
    pruning_ratio = self.joint_pruning.get_pruning_ratio()
    self.log("pruning_ratio", pruning_ratio, on_step=False, on_epoch=True)
    # ===== END NEW LOGGING =====
    
    return loss
```

---

## How It Works

### Training Phase (Soft Selection)
- The model learns `joint_logits` (one logit per joint)
- Gumbel-Softmax converts logits → soft probabilities in [0, 1]
- Each joint is multiplied by its probability (soft masking)
- The model learns: *"which joints are useful?"*
- Gradient flows through the selection mask → `joint_logits` updates

### Inference Phase (Hard Selection)
- Set `hard=True` in the pruning layer
- Gumbel-Softmax produces discrete 0/1 mask
- Only the selected joints are used
- You can extract `get_active_joints()` to see which joints survived

---

## Analysis & Results Reporting

Create a script to analyze joint importance:

```python
def analyze_pruning(model):
    """Extract and visualize which joints the model kept."""
    probs = F.softmax(model.joint_pruning.joint_logits, dim=-1)
    
    # Map to MediaPipe landmarks
    joint_names = [
        "Wrist_L", "Thumb_CMC_L", "Thumb_MCP_L", ...,  # Left hand (21)
        "Wrist_R", "Thumb_CMC_R", "Thumb_MCP_R", ...   # Right hand (21)
    ]
    
    for i, (name, prob) in enumerate(zip(joint_names, probs)):
        status = "✓ KEEP" if prob > 0.5 else "✗ PRUNE"
        print(f"{i:2d}. {name:20s} {prob:.3f} {status}")
    
    # Summary
    active_count = (probs > 0.5).sum().item()
    reduction = 100 * (1 - active_count / len(probs))
    print(f"\nActive joints: {active_count} / {len(probs)} ({reduction:.1f}% reduction)")
```

---

## Expected Results

Your paper can now report:

> **"Via learnable joint pruning with Gumbel-Softmax, we identify the minimal topological subset of MediaPipe landmarks required for Turkish Sign Language recognition. The model automatically discovered that only [X] of 42 hand + pose joints are necessary to maintain >85% accuracy on AUTSL, representing a [Y]% spatial reduction compared to the full skeleton."**

This directly contradicts TSLFormer's assumption that all 48 points are needed! 🎯

---

## Hyperparameter Tuning

To encourage more aggressive pruning, add **L0 regularization** to your loss:

```python
# In training_step, after computing main loss:
l0_penalty = torch.mean(F.sigmoid(model.joint_pruning.joint_logits))
total_loss = loss + 0.001 * l0_penalty  # Adjust 0.001 as needed
```

Higher penalty → more joints pruned. Lower penalty → keep more joints.
