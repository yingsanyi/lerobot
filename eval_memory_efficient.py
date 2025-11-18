#!/usr/bin/env python3
"""
Memory-efficient evaluation: Load policy first, then create minimal environments
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MUJOCO_GL'] = 'egl'

import sys
import torch
import gc

sys.path.insert(0, '/media/ai-robot/hometwo/wyl/lerobot/v.0.4.0/lerobot/src')

torch.cuda.empty_cache()
gc.collect()
print(f"[1/4] Initial GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

from lerobot.policies.pi05.modeling_pi05 import PI05Policy

print("\n[2/4] Loading policy from checkpoint...")
policy = PI05Policy.from_pretrained("lerobot/pi05_libero_finetuned")
policy.eval()
device = policy.config.device
print(f"✓ Policy loaded | GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB | Device: {device}")

# Check if policy has a tokenizer
print(f"Policy has tokenizer: {hasattr(policy, 'tokenizer')}")
if hasattr(policy, 'tokenizer'):
    print(f"Tokenizer type: {type(policy.tokenizer)}")

print("\n[3/4] Creating single LIBERO environment (task 0 only)...")
from lerobot.envs.libero import create_libero_envs
import gymnasium as gym

envs = create_libero_envs(
    task="libero_spatial",
    n_envs=1,
    gym_kwargs={"task_ids": [0]},
    camera_name="agentview_image,robot0_eye_in_hand_image",
    init_states=False,
    env_cls=gym.vector.SyncVectorEnv,
)

print(f"✓ Environment created | GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

task_0_env = envs["libero_spatial"][0]

# Get task description
from libero.libero.benchmark import get_benchmark_dict
benchmark_dict = get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
task_description = task.language

print(f"\nTask 0 description: '{task_description}'")

# Use policy's tokenizer or create one
if hasattr(policy, 'tokenizer') and policy.tokenizer is not None:
    tokenizer = policy.tokenizer
    print("Using policy's built-in tokenizer")
else:
    # Try to load from the policy's pretrained path
    from transformers import AutoTokenizer
    print("Loading tokenizer from policy pretrained path...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(policy.config.pretrained_path))
    except:
        # Use Gemma tokenizer as fallback (not gated)
        print("Loading Gemma tokenizer as fallback...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

tokenized = tokenizer(
    task_description,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=policy.config.tokenizer_max_length,
)

language_tokens = tokenized["input_ids"].to(device)
language_mask = tokenized["attention_mask"].to(device)

print(f"Language tokens shape: {language_tokens.shape}")

print("\n[4/4] Running evaluation...")
from tqdm import tqdm
import numpy as np

success_count = 0
n_episodes = 1

for episode in range(n_episodes):
    obs, info = task_0_env.reset()
    done = False
    step = 0
    episode_reward = 0
    
    print(f"Starting episode {episode+1}")
    
    with tqdm(total=520, desc=f"Episode {episode+1}/{n_episodes}") as pbar:
        while not done and step < 520:
            # Build observation dict
            obs_dict = {}
            
            # Add language tokens (constant for the episode)
            obs_dict["observation.language.tokens"] = language_tokens
            obs_dict["observation.language.attention_mask"] = language_mask
            
            # Extract images from pixels dict
            if "pixels" in obs and isinstance(obs["pixels"], dict):
                pixel_dict = obs["pixels"]
                
                # Process 'image' (agentview)
                if "image" in pixel_dict:
                    img = pixel_dict["image"]
                    if len(img.shape) == 4:  # Batched: (1, H, W, C)
                        img = img[0]  # -> (H, W, C)
                    
                    # Convert to tensor: (H, W, C) -> (1, C, H, W)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    obs_dict["observation.images.image"] = img_tensor
                
                # Process 'image2' (wrist camera)
                if "image2" in pixel_dict:
                    img2 = pixel_dict["image2"]
                    if len(img2.shape) == 4:
                        img2 = img2[0]
                    
                    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    obs_dict["observation.images.image2"] = img2_tensor
            
            # Look for state
            for key in obs.keys():
                if key not in ["pixels"]:
                    val = obs[key]
                    if hasattr(val, 'shape') and len(val.shape) >= 1:
                        state = val
                        if len(state.shape) == 2:
                            state = state[0]
                        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
                        obs_dict["observation.state"] = state_tensor
                        break
            
            # If no state found, create dummy state
            if "observation.state" not in obs_dict:
                obs_dict["observation.state"] = torch.zeros(1, 8).float().to(device)
            
            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(obs_dict)
            
            # Step environment
            action_np = action.cpu().numpy()
            if len(action_np.shape) == 1:
                action_np = action_np[None, :]
            
            obs, reward, terminated, truncated, info = task_0_env.step(action_np)
            
            # Handle vectorized outputs
            if isinstance(terminated, np.ndarray):
                done = terminated[0] or truncated[0]
                reward = reward[0] if isinstance(reward, np.ndarray) else reward
            else:
                done = terminated or truncated
            
            episode_reward += reward
            step += 1
            pbar.update(1)
            
            # Get success from info
            success = False
            if isinstance(info, dict):
                if "final_info" in info and len(info["final_info"]) > 0 and info["final_info"][0] is not None:
                    success = info["final_info"][0].get("success", False)
                elif "success" in info:
                    success = info["success"][0] if isinstance(info["success"], np.ndarray) else info["success"]
            
            pbar.set_postfix({"reward": f"{episode_reward:.3f}", "step": step, "success": success})
            
            if done:
                break
    
    # Get final success status
    success = False
    if isinstance(info, dict):
        if "final_info" in info and len(info["final_info"]) > 0 and info["final_info"][0] is not None:
            success = info["final_info"][0].get("success", False)
        elif "success" in info:
            success = info["success"][0] if isinstance(info["success"], np.ndarray) else info["success"]
    
    if success:
        success_count += 1
    
    print(f"\nEpisode {episode+1}: {'✅ SUCCESS' if success else '❌ FAILED'} | Steps: {step} | Reward: {episode_reward:.3f}")

print(f"\n{'='*60}")
print(f"Final Results: {success_count}/{n_episodes} successful ({100*success_count/n_episodes:.1f}%)")
print(f"Final GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"{'='*60}")

task_0_env.close()
