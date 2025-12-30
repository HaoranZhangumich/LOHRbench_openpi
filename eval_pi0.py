#!/usr/bin/env python3
"""
Evaluate Pi0-FAST finetuned with LoRA on LoHRBench RLDS dataset.

This script:
- Loads Pi0-FAST checkpoint with LoRA from training
- Evaluates on RLDS TFDS episodes
- Compares predicted actions to ground truth
- Generates plots and metrics

Example:
python eval_pi0.py \
  --checkpoint_dir /home/haoran-zhang/pi0.5_pi0/LOHRbench_openpi/third_party/openpi/checkpoints/pi0_lohrbench_rlds_finetune/lohrbench_exp_005/70000 \
  --config_name pi0_lohrbench_rlds_finetune \
  --builder_dir /home/haoran-zhang/data/Lohrbench_rlds/lohrbench_rlds/lohrbench_rlds/0.1.0 \
  --split train \
  --episode_index 0 \
  --num_episodes 1 \
  --max_steps 100 \
  --out_dir ./eval_results \
  --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import matplotlib.pyplot as plt

# Disable TensorFlow GPU usage (only for data loading)
tf.config.set_visible_devices([], "GPU")

# Add OpenPI to path
OPENPI_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(OPENPI_ROOT))

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import array_typing as at


# =============================================================================
# Helper functions
# =============================================================================
def decode_text(x: Any) -> str:
    """Decode bytes to string."""
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def uint8_image(x: Any) -> np.ndarray:
    """Convert to uint8 image."""
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def pack_state_from_qpos(qpos_9: Any) -> np.ndarray:
    """Convert 9D qpos to 8D state: [qpos[:7], mean(qpos[7:9])]."""
    qpos_9 = np.asarray(qpos_9, dtype=np.float32)
    if qpos_9.shape[-1] != 9:
        raise ValueError(f"Expected qpos last dim=9, got {qpos_9.shape}")
    joints7 = qpos_9[..., :7]
    gripper1 = np.mean(qpos_9[..., 7:9], axis=-1, keepdims=True)
    return np.concatenate([joints7, gripper1], axis=-1).astype(np.float32)


def summarize_vec(name: str, x: np.ndarray) -> None:
    """Print vector statistics."""
    x = np.asarray(x)
    print(
        f"{name}: shape={x.shape} min={x.min():.4f} max={x.max():.4f} "
        f"mean={x.mean():.4f} std={x.std():.4f}",
        flush=True,
    )


def save_images(
    base_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
    out_dir: str,
    stride: int = 50,
    max_frames: int = 30,
):
    """Save sampled frames as images."""
    os.makedirs(out_dir, exist_ok=True)
    T = min(len(base_frames), len(wrist_frames))
    idxs = list(range(0, T, max(1, stride)))[:max_frames]
    
    for t in idxs:
        Image.fromarray(base_frames[t]).save(
            os.path.join(out_dir, f"base_t{t:04d}.png")
        )
        Image.fromarray(wrist_frames[t]).save(
            os.path.join(out_dir, f"wrist_t{t:04d}.png")
        )
    print(f"✅ Saved {len(idxs)} image pairs to: {out_dir}")


def plot_debug(gt: np.ndarray, pred: np.ndarray, out_dir: str):
    """Generate comparison plots."""
    os.makedirs(out_dir, exist_ok=True)
    T, D = gt.shape
    x = np.arange(T)

    # Per-dimension plots
    for d in range(D):
        plt.figure(figsize=(10, 5))
        plt.plot(x, gt[:, d], label="Ground Truth", linewidth=2)
        plt.plot(x, pred[:, d], linestyle="--", label="Predicted", linewidth=2)
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel(f"Action Dimension {d}", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"dim_{d:02d}.png"), dpi=120)
        plt.close()

    # L2 error plot
    plt.figure(figsize=(10, 5))
    l2_err = np.linalg.norm(pred - gt, axis=1)
    plt.plot(x, l2_err, linewidth=2, color='red')
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("L2 Error", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "l2_error.png"), dpi=120)
    plt.close()

    print(f"✅ Saved plots to: {out_dir}")


def print_error_summary(gt_arr: np.ndarray, pred_arr: np.ndarray) -> None:
    """Print detailed error metrics."""
    err = (pred_arr - gt_arr).astype(np.float64)
    abs_err = np.abs(err)
    mse = err ** 2
    mae = abs_err.mean(axis=0)
    rmse = np.sqrt(mse.mean(axis=0))
    
    print(f"\n{'='*60}")
    print(f"ERROR METRICS")
    print(f"{'='*60}")
    print(f"Mean MAE:  {mae.mean():.6f}")
    print(f"Mean RMSE: {rmse.mean():.6f}")
    print(f"Max |err|: {abs_err.max():.6f}")
    print(f"\nPer-dimension breakdown:")
    for d in range(gt_arr.shape[1]):
        dim_name = f"Joint {d}" if d < 7 else "Gripper"
        print(f"  {dim_name:10s}: MAE={mae[d]:.6f}  RMSE={rmse[d]:.6f}  Max={abs_err[:, d].max():.6f}")
    print(f"{'='*60}\n")


# =============================================================================
# Model loading using official OpenPI API
# =============================================================================
def load_policy(checkpoint_dir: Path, config: _config.TrainConfig):
    """
    Load Pi0-FAST policy with LoRA using official OpenPI API.
    
    This function uses policy_config.create_trained_policy() which:
    - Automatically loads the model based on the config
    - Handles checkpoint restoration (including LoRA weights)
    - Sets up all necessary transforms for inference
    - Returns a ready-to-use policy object
    """
    print(f"\n🤖 Loading policy from: {checkpoint_dir}")
    print(f"   Config: {config.name}")
    print(f"   Model type: {config.model.__class__.__name__}")
    print(f"   Action dim: {config.model.action_dim}")
    print(f"   Action horizon: {config.model.action_horizon}")
    
    # Use the official OpenPI API to create trained policy
    # This handles all model loading, checkpoint restoration, and transform setup
    policy = policy_config.create_trained_policy(
        train_config=config,
        checkpoint_dir=checkpoint_dir,
    )
    
    print(f"✅ Policy loaded successfully")
    return policy


def prepare_observation_dict(
    base_img: np.ndarray,
    wrist_img: np.ndarray,
    state: np.ndarray,
    instruction: str,
) -> dict:
    """
    Prepare observation dict in the format expected by your policy.
    
    Adjust the keys based on your policy's InputTransform class.
    For LoHRBench, you likely have something like:
    - base_rgb -> primary camera
    - hand_rgb -> wrist camera
    - qpos state -> proprioceptive state
    """
    # This format should match what your policy expects
    # Adjust the keys based on your Inputs transform class
    obs = {
        "observation/image": base_img,
        "observation/wrist_image": wrist_img,
        "observation/qpos": state,
        "prompt": instruction,
    }
    return obs


# =============================================================================
# Main evaluation
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Pi0-FAST on LoHRBench")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Path to checkpoint directory (e.g., checkpoints/.../50000)")
    parser.add_argument("--config_name", type=str, required=True,
                       help="Training config name (e.g., pi0_lohrbench_rlds_finetune)")
    parser.add_argument("--builder_dir", type=str, required=True,
                       help="Path to RLDS dataset builder directory")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to evaluate")
    parser.add_argument("--episode_index", type=int, default=0,
                       help="Starting episode index")
    parser.add_argument("--num_episodes", type=int, default=5,
                       help="Number of episodes to evaluate")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="Max steps per episode")
    parser.add_argument("--out_dir", type=str, default="./eval_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--save_images", action="store_true",
                       help="Save episode images")
    parser.add_argument("--img_stride", type=int, default=50,
                       help="Stride for saving images")
    parser.add_argument("--img_max_frames", type=int, default=30,
                       help="Max frames to save")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 80)
    print("Pi0-FAST Evaluation on LoHRBench")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Config: {args.config_name}")
    print(f"Dataset: {args.builder_dir}")
    print(f"Split: {args.split}")
    print(f"Episodes: {args.episode_index} to {args.episode_index + args.num_episodes - 1}")
    print(f"Output: {args.out_dir}")
    print("=" * 80)
    
    # Load config
    print("\n📋 Loading training config...")
    config = _config.get_config(args.config_name)
    
    # Load policy using official OpenPI API
    # This handles model initialization, checkpoint loading, and LoRA restoration
    policy = load_policy(checkpoint_dir, config)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {args.builder_dir}")
    builder = tfds.builder_from_directory(args.builder_dir)
    ds = builder.as_dataset(split=args.split, shuffle_files=False)
    print(f"✅ Dataset loaded")
    
    all_gt = []
    all_pred = []
    
    # Evaluate episodes
    for ep_idx in range(args.episode_index, args.episode_index + args.num_episodes):
        print(f"\n{'='*80}")
        print(f"Episode {ep_idx}")
        print(f"{'='*80}")
        
        # Get episode
        ep = next(iter(ds.skip(ep_idx).take(1)))
        steps_list = list(ep["steps"].as_numpy_iterator())
        
        if len(steps_list) < 2:
            print(f"⚠️  Episode too short ({len(steps_list)} steps). Skipping.")
            continue
        
        # Remove last step (often a dummy/terminal step)
        steps_list = steps_list[:-1]
        print(f"Episode length: {len(steps_list)} steps")
        
        # Get instruction
        instruction = decode_text(steps_list[0]["language_instruction"])
        print(f"Instruction: '{instruction}'")
        
        gt_list = []
        pred_list = []
        base_frames = []
        wrist_frames = []
        
        T = min(len(steps_list), args.max_steps)
        
        for t in range(T):
            if t % 20 == 0:
                print(f"  Step {t}/{T}...", end="\r")
            
            step_data = steps_list[t]
            obs_raw = step_data["observation"]
            
            # Get images
            base_img = uint8_image(obs_raw["base_rgb"])
            wrist_img = uint8_image(obs_raw["hand_rgb"])
            
            if args.save_images:
                base_frames.append(base_img)
                wrist_frames.append(wrist_img)
            
            # Get state
            state = pack_state_from_qpos(obs_raw["qpos"])  # (8,)
            
            # Get ground truth action
            gt_action = np.asarray(step_data["action"], dtype=np.float32)
            if gt_action.ndim == 2:  # (action_horizon, action_dim)
                gt_action = gt_action[0]  # Take first action
            elif gt_action.shape[-1] != 8:
                raise ValueError(f"Unexpected action shape: {gt_action.shape}")
            
            # Prepare observation dict
            obs = prepare_observation_dict(
                base_img, wrist_img, state, instruction
            )
            
            # Run inference using the policy's infer method
            # The policy handles all transforms internally
            result = policy.infer(obs)
            pred_actions = result["action"]  # Shape: (action_horizon, action_dim)
            
            # Take the first action from the action chunk
            pred_action = np.array(pred_actions[0, :])
            
            # Log first step
            if t == 0:
                print(f"\n  First step sanity check:")
                summarize_vec("    GT action", gt_action)
                summarize_vec("    Pred action", pred_action)
                print(f"    GT:   {np.round(gt_action, 3)}")
                print(f"    Pred: {np.round(pred_action, 3)}")
                print(f"    |Err|: {np.round(np.abs(pred_action - gt_action), 3)}")
            
            gt_list.append(gt_action)
            pred_list.append(pred_action)
        
        print(f"  Step {T}/{T}... Done!")
        
        # Convert to arrays
        gt_arr = np.stack(gt_list, axis=0)
        pred_arr = np.stack(pred_list, axis=0)
        
        # Save results
        ep_dir = os.path.join(args.out_dir, f"episode_{ep_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        
        np.save(os.path.join(ep_dir, "gt_actions.npy"), gt_arr)
        np.save(os.path.join(ep_dir, "pred_actions.npy"), pred_arr)
        
        # Generate plots
        plot_debug(gt_arr, pred_arr, os.path.join(ep_dir, "plots"))
        
        # Save images if requested
        if args.save_images:
            save_images(
                base_frames, wrist_frames,
                os.path.join(ep_dir, "images"),
                stride=args.img_stride,
                max_frames=args.img_max_frames,
            )
        
        # Print episode metrics
        print(f"\nEpisode {ep_idx} metrics:")
        print_error_summary(gt_arr, pred_arr)
        
        all_gt.append(gt_arr)
        all_pred.append(pred_arr)
    
    # Aggregate metrics
    if all_gt:
        all_gt = np.concatenate(all_gt, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)
        
        print(f"\n{'='*80}")
        print(f"AGGREGATE METRICS ({all_gt.shape[0]} total steps)")
        print(f"{'='*80}")
        print_error_summary(all_gt, all_pred)
        
        # Save aggregate results
        np.save(os.path.join(args.out_dir, "all_gt_actions.npy"), all_gt)
        np.save(os.path.join(args.out_dir, "all_pred_actions.npy"), all_pred)
    
    print(f"\n✅ Evaluation complete! Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()