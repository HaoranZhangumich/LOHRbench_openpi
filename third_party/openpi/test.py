# # Comprehensive dataset verification script
# import jax
# import numpy as np
# from openpi.training import config as _config
# from openpi.training import data_loader as _data_loader

# # Load your config
# config = _config.get_config("pi0_lohrbench_rlds_finetune")

# # Create data loader with just a few batches for testing
# data_loader = _data_loader.create_data_loader(
#     config,
#     shuffle=False,
#     num_batches=5,
#     skip_norm_stats=True,  # Skip normalization to see raw values
# )

# print("=" * 80)
# print("COMPREHENSIVE DATASET VERIFICATION")
# print("=" * 80)

# all_issues = []

# # Iterate through batches
# for i, (observation, actions) in enumerate(data_loader):
#     print(f"\n{'='*80}")
#     print(f"BATCH {i+1}")
#     print(f"{'='*80}")
    
#     batch_issues = []
    
#     # ========== Check Images ==========
#     print("\n[IMAGES]")
#     if hasattr(observation, 'images') and observation.images:
#         for img_name, img in observation.images.items():
#             print(f"  {img_name}:")
#             print(f"    Shape: {img.shape}, Dtype: {img.dtype}")
#             print(f"    Range: [{img.min():.2f}, {img.max():.2f}], Mean: {img.mean():.2f}")
            
#             # Check for NaN/Inf
#             if np.any(np.isnan(img)):
#                 issue = f"    ⚠️  WARNING: {img_name} contains NaN values!"
#                 print(issue)
#                 batch_issues.append(issue)
#             if np.any(np.isinf(img)):
#                 issue = f"    ⚠️  WARNING: {img_name} contains Inf values!"
#                 print(issue)
#                 batch_issues.append(issue)
            
#             # Check for all-zero images (padding)
#             num_zero_images = np.sum(np.all(img == 0, axis=(1, 2, 3)))
#             if num_zero_images > 0:
#                 issue = f"    ⚠️  WARNING: {num_zero_images}/{img.shape[0]} images are all zeros (padding)!"
#                 print(issue)
#                 batch_issues.append(issue)
            
#             # Check for suspiciously low variance (might be corrupted/padding)
#             variances = np.var(img, axis=(1, 2, 3))
#             low_var_count = np.sum(variances < 0.01)
#             if low_var_count > img.shape[0] * 0.1:  # More than 10% have low variance
#                 issue = f"    ⚠️  WARNING: {low_var_count}/{img.shape[0]} images have very low variance (possibly corrupted)!"
#                 print(issue)
#                 batch_issues.append(issue)
            
#             print(f"    ✓ No padding/empty images detected" if num_zero_images == 0 else "")
    
#     # ========== Check Image Masks ==========
#     print("\n[IMAGE MASKS]")
#     if hasattr(observation, 'image_masks') and observation.image_masks:
#         for mask_name, mask in observation.image_masks.items():
#             mask_array = np.asarray(mask)
#             true_count = np.sum(mask_array == True)
#             false_count = np.sum(mask_array == False)
#             print(f"  {mask_name}: {true_count} real, {false_count} padding")
            
#             if false_count > 0:
#                 issue = f"    ⚠️  WARNING: {mask_name} has {false_count} padded images!"
#                 print(issue)
#                 batch_issues.append(issue)
    
#     # ========== Check State ==========
#     print("\n[STATE]")
#     if hasattr(observation, 'state') and observation.state is not None:
#         state = observation.state
#         print(f"  Shape: {state.shape}, Dtype: {state.dtype}")
#         print(f"  Range: [{state.min():.3f}, {state.max():.3f}], Mean: {state.mean():.3f}")
        
#         # Check for NaN/Inf
#         if np.any(np.isnan(state)):
#             issue = f"  ⚠️  WARNING: State contains NaN values!"
#             print(issue)
#             batch_issues.append(issue)
#         if np.any(np.isinf(state)):
#             issue = f"  ⚠️  WARNING: State contains Inf values!"
#             print(issue)
#             batch_issues.append(issue)
        
#         # Check for all-zero states (padding)
#         num_zero_states = np.sum(np.all(state == 0, axis=1))
#         if num_zero_states > 0:
#             issue = f"  ⚠️  WARNING: {num_zero_states}/{state.shape[0]} states are all zeros (padding)!"
#             print(issue)
#             batch_issues.append(issue)
        
#         # Check individual dimensions
#         print(f"  Per-dimension stats:")
#         for dim in range(min(state.shape[1], 8)):  # Show first 8 dims
#             dim_values = state[:, dim]
#             num_zeros = np.sum(dim_values == 0)
#             print(f"    Dim {dim}: range=[{dim_values.min():.3f}, {dim_values.max():.3f}], "
#                   f"std={dim_values.std():.3f}, zeros={num_zeros}/{len(dim_values)}")
        
#         print(f"  ✓ No padding/empty states detected" if num_zero_states == 0 else "")
    
#     # ========== Check Actions ==========
#     print("\n[ACTIONS]")
#     print(f"  Shape: {actions.shape}, Dtype: {actions.dtype}")
#     print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}], Mean: {actions.mean():.3f}")
    
#     # Check for NaN/Inf
#     if np.any(np.isnan(actions)):
#         issue = f"  ⚠️  WARNING: Actions contain NaN values!"
#         print(issue)
#         batch_issues.append(issue)
#     if np.any(np.isinf(actions)):
#         issue = f"  ⚠️  WARNING: Actions contain Inf values!"
#         print(issue)
#         batch_issues.append(issue)
    
#     # Check for all-zero action sequences (padding)
#     num_zero_actions = np.sum(np.all(actions == 0, axis=(1, 2)))
#     if num_zero_actions > 0:
#         issue = f"  ⚠️  WARNING: {num_zero_actions}/{actions.shape[0]} action sequences are all zeros!"
#         print(issue)
#         batch_issues.append(issue)
    
#     # Check per action dimension
#     print(f"  Per-dimension stats (first timestep):")
#     for dim in range(actions.shape[2]):
#         dim_values = actions[:, 0, dim]  # First timestep
#         num_zeros = np.sum(dim_values == 0)
#         print(f"    Dim {dim}: range=[{dim_values.min():.3f}, {dim_values.max():.3f}], "
#               f"std={dim_values.std():.3f}, zeros={num_zeros}/{len(dim_values)}")
    
#     print(f"  ✓ No padding/empty actions detected" if num_zero_actions == 0 else "")
    
#     # ========== Check Prompts ==========
#     print("\n[PROMPTS]")
#     # Check for raw prompts (before tokenization)
#     if hasattr(observation, 'prompt') and observation.prompt is not None:
#         prompts = observation.prompt
#         print(f"  Raw prompts found: {type(prompts)}")
#         if isinstance(prompts, list):
#             print(f"  Count: {len(prompts)} prompts")
#             print(f"  Sample prompts (first 3): {prompts[:3]}")
#     # Check for tokenized prompts (after model transforms)
#     elif hasattr(observation, 'tokenized_prompt') and observation.tokenized_prompt is not None:
#         tokenized = observation.tokenized_prompt
#         print(f"  ✓ Prompts have been tokenized")
#         print(f"  Tokenized prompt shape: {tokenized.shape}, dtype: {tokenized.dtype}")
#         print(f"  Token range: [{tokenized.min()}, {tokenized.max()}]")
#         print(f"  Sample tokens (first sample, first 20): {tokenized[0, :20]}")
        
#         # Check for prompt mask
#         if hasattr(observation, 'tokenized_prompt_mask'):
#             mask = observation.tokenized_prompt_mask
#             print(f"  Prompt mask shape: {mask.shape}")
#             print(f"  Active tokens per sample (mean): {mask.sum(axis=1).mean():.1f}")
#     else:
#         issue = f"  ⚠️  WARNING: No prompts (raw or tokenized) found in observation!"
#         print(issue)
#         batch_issues.append(issue)
        
#         # ========== Batch Summary ==========
#         if batch_issues:
#             print(f"\n{'!'*80}")
#             print(f"BATCH {i+1} ISSUES SUMMARY: {len(batch_issues)} issues found")
#             print(f"{'!'*80}")
#             for issue in batch_issues:
#                 print(issue)
#             all_issues.extend(batch_issues)
#         else:
#             print(f"\n{'✓'*80}")
#             print(f"BATCH {i+1}: ALL CHECKS PASSED - No issues detected!")
#             print(f"{'✓'*80}")

# # ========== Final Summary ==========
# print("\n" + "=" * 80)
# print("FINAL VERIFICATION SUMMARY")
# print("=" * 80)

# if all_issues:
#     print(f"\n❌ FOUND {len(all_issues)} TOTAL ISSUES:")
#     for idx, issue in enumerate(all_issues, 1):
#         print(f"{idx}. {issue}")
#     print("\n⚠️  Please review and fix these issues before training!")
# else:
#     print("\n✅ ALL CHECKS PASSED!")
#     print("✅ Dataset is clean - no padding, empty values, or NaN/Inf detected")
#     print("✅ Ready for training!")

# print("\n" + "=" * 80)

#!/usr/bin/env python3
"""
Comprehensive LOHRbench Training Debug Script
Checks for common issues that cause loss plateau at ~0.5
"""

import jax
import numpy as np
import pathlib
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

print("=" * 100)
print("LOHRbench TRAINING DEBUG SCRIPT")
print("=" * 100)

# ============================================================================
# SECTION 1: CONFIG INSPECTION
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 1: TRAINING CONFIG INSPECTION")
print("=" * 100)

config = _config.get_config("pi0_lohrbench_rlds_finetune")

print(f"\n[MODEL CONFIG]")
print(f"  Model type: {config.model.model_type}")
print(f"  Action dim: {config.model.action_dim}")
print(f"  Action horizon: {config.model.action_horizon}")
print(f"  Max token len: {config.model.max_token_len}")
if hasattr(config.model, 'paligemma_variant'):
    print(f"  PaliGemma variant: {config.model.paligemma_variant}")

print(f"\n[TRAINING CONFIG]")
print(f"  Batch size: {config.batch_size}")
print(f"  Num train steps: {config.num_train_steps}")
print(f"  Num workers: {config.num_workers}")
print(f"  EMA decay: {config.ema_decay}")

print(f"\n[LEARNING RATE SCHEDULE] 🔥 CRITICAL")
lr_schedule = config.lr_schedule
print(f"  Schedule type: {type(lr_schedule).__name__}")
if hasattr(lr_schedule, 'warmup_steps'):
    print(f"  Warmup steps: {lr_schedule.warmup_steps}")
if hasattr(lr_schedule, 'peak_lr'):
    print(f"  Peak LR: {lr_schedule.peak_lr}")
if hasattr(lr_schedule, 'decay_steps'):
    print(f"  Decay steps: {lr_schedule.decay_steps}")
if hasattr(lr_schedule, 'decay_lr'):
    print(f"  Decay LR: {lr_schedule.decay_lr}")

# ⚠️ CHECK FOR ISSUE #1: LR schedule vs num_train_steps mismatch
if hasattr(lr_schedule, 'decay_steps') and hasattr(lr_schedule, 'decay_lr'):
    if lr_schedule.decay_steps < config.num_train_steps:
        print(f"\n  ❌ CRITICAL ISSUE FOUND!")
        print(f"     LR will decay to {lr_schedule.decay_lr} after {lr_schedule.decay_steps} steps")
        print(f"     But training runs for {config.num_train_steps} steps!")
        print(f"     This means {config.num_train_steps - lr_schedule.decay_steps} steps at low LR = {lr_schedule.decay_lr}")
        print(f"  🔧 FIX: Set decay_steps={config.num_train_steps} in config")
    else:
        print(f"  ✓ LR schedule matches training length")

print(f"\n[DATA CONFIG]")
print(f"  Repo ID: {config.data.repo_id}")
if hasattr(config.data, 'rlds_data_dir'):
    print(f"  RLDS data dir: {config.data.rlds_data_dir}")

# ============================================================================
# SECTION 2: NORM STATS CHECK
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 2: NORMALIZATION STATISTICS CHECK")
print("=" * 100)

data_config = config.data.create(config.assets_dirs, config.model)

print(f"\n[NORM STATS LOCATION]")
print(f"  Assets dir: {config.assets_dirs}")
print(f"  Expected norm stats path: {config.assets_dirs / (data_config.asset_id or 'unknown')}")

if data_config.norm_stats is None:
    print(f"\n  ❌ CRITICAL ISSUE: Norm stats are NOT loaded!")
    print(f"  🔧 FIX: Run this command first:")
    print(f"     uv run --group rlds scripts/compute_norm_stats.py \\")
    print(f"         --config-name pi0_lohrbench_rlds_finetune \\")
    print(f"         --max-frames 100000")
else:
    print(f"\n  ✓ Norm stats loaded successfully!")
    print(f"\n  [NORM STATS DETAILS]")
    from openpi.transforms import flatten_dict
    flat_stats = flatten_dict(data_config.norm_stats)
    for key, stats in flat_stats.items():
        print(f"    {key}:")
        print(f"      Mean shape: {stats.mean.shape}, range: [{stats.mean.min():.3f}, {stats.mean.max():.3f}]")
        print(f"      Std shape: {stats.std.shape}, range: [{stats.std.min():.3f}, {stats.std.max():.3f}]")
        if stats.q01 is not None:
            print(f"      Q01 shape: {stats.q01.shape}")
            print(f"      Q99 shape: {stats.q99.shape}")

# ============================================================================
# SECTION 3: DATA LOADER INSPECTION (RAW DATA)
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 3: RAW DATA INSPECTION (Before Normalization)")
print("=" * 100)

print("\n⏳ Loading raw data (first 3 batches, no normalization)...")

raw_data_loader = _data_loader.create_data_loader(
    config,
    shuffle=False,
    num_batches=3,
    skip_norm_stats=True,  # Skip normalization to see raw values
)

for batch_idx, (observation, actions) in enumerate(raw_data_loader):
    print(f"\n{'─'*100}")
    print(f"RAW BATCH {batch_idx + 1}/3")
    print(f"{'─'*100}")
    
    # ========== PROMPTS CHECK ==========
    print(f"\n[PROMPTS] 🔥 CRITICAL CHECK")
    
    # Try to find prompts in various places
    prompts_found = False
    
    # Check in observation attributes
    for attr_name in ['prompt', 'prompts', 'tokenized_prompt', 'language_instruction']:
        if hasattr(observation, attr_name):
            attr_value = getattr(observation, attr_name)
            if attr_value is not None:
                prompts_found = True
                print(f"  Found '{attr_name}': {type(attr_value)}")
                
                if isinstance(attr_value, list):
                    print(f"    Type: List with {len(attr_value)} elements")
                    print(f"    First 3 prompts:")
                    for i, p in enumerate(attr_value[:3]):
                        print(f"      [{i}] {repr(p)[:100]}")
                    
                    # Check for duplicates
                    unique_prompts = set(attr_value)
                    if len(unique_prompts) < len(attr_value):
                        print(f"    ℹ️  Found {len(unique_prompts)} unique prompts out of {len(attr_value)} (some duplication)")
                    else:
                        print(f"    ✓ All {len(attr_value)} prompts are unique")
                
                elif isinstance(attr_value, np.ndarray):
                    print(f"    Type: Array, shape: {attr_value.shape}, dtype: {attr_value.dtype}")
                    if attr_value.dtype == np.int32 or attr_value.dtype == np.int64:
                        print(f"    Range: [{attr_value.min()}, {attr_value.max()}]")
                        print(f"    First sample tokens (first 30): {attr_value[0, :30]}")
                    else:
                        print(f"    First 3 values: {attr_value[:3]}")
                else:
                    print(f"    Value: {repr(attr_value)[:200]}")
    
    if not prompts_found:
        print(f"  ❌ WARNING: No prompts found in observation!")
        print(f"  Available observation attributes: {[a for a in dir(observation) if not a.startswith('_')]}")
    
    # ========== STATE CHECK ==========
    print(f"\n[STATE/QPOS] 🔥 ACTION DIMENSION CHECK")
    if hasattr(observation, 'state') and observation.state is not None:
        state = observation.state
        print(f"  State shape: {state.shape} (batch_size, state_dim)")
        print(f"  Expected: ({config.batch_size}, {config.model.action_dim})")
        
        if state.shape[1] != config.model.action_dim:
            print(f"  ❌ DIMENSION MISMATCH!")
            print(f"     State has {state.shape[1]} dims but model expects {config.model.action_dim} dims")
        else:
            print(f"  ✓ State dimensions match model")
        
        # Show first sample in detail
        print(f"\n  First sample state (all {state.shape[1]} dimensions):")
        first_state = state[0]
        for dim in range(state.shape[1]):
            print(f"    Dim {dim:2d}: {first_state[dim]:8.4f}")
        
        # Statistics
        print(f"\n  Per-dimension statistics:")
        print(f"  {'Dim':<4} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Zeros':>8}")
        print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for dim in range(state.shape[1]):
            dim_values = state[:, dim]
            num_zeros = np.sum(np.abs(dim_values) < 1e-6)
            print(f"  {dim:<4d} {dim_values.min():10.4f} {dim_values.max():10.4f} "
                  f"{dim_values.mean():10.4f} {dim_values.std():10.4f} {num_zeros:>8d}/{len(dim_values)}")
    
    # ========== ACTIONS CHECK ==========
    print(f"\n[ACTIONS]")
    print(f"  Actions shape: {actions.shape} (batch_size, action_horizon, action_dim)")
    print(f"  Expected: ({config.batch_size}, {config.model.action_horizon}, {config.model.action_dim})")
    
    if actions.shape[1] != config.model.action_horizon:
        print(f"  ❌ Action horizon mismatch! {actions.shape[1]} vs {config.model.action_horizon}")
    if actions.shape[2] != config.model.action_dim:
        print(f"  ❌ Action dim mismatch! {actions.shape[2]} vs {config.model.action_dim}")
    
    # Show first sample, first 3 timesteps
    print(f"\n  First sample actions (first 3 timesteps of {actions.shape[1]}):")
    for t in range(min(3, actions.shape[1])):
        print(f"    t={t}: {actions[0, t, :]}")
    
    # Statistics for first timestep
    print(f"\n  Per-dimension statistics (first timestep only):")
    print(f"  {'Dim':<4} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Zeros':>8}")
    print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for dim in range(actions.shape[2]):
        dim_values = actions[:, 0, dim]  # First timestep
        num_zeros = np.sum(np.abs(dim_values) < 1e-6)
        print(f"  {dim:<4d} {dim_values.min():10.4f} {dim_values.max():10.4f} "
              f"{dim_values.mean():10.4f} {dim_values.std():10.4f} {num_zeros:>8d}/{len(dim_values)}")
    
    # Check for identical actions across timesteps (might indicate an issue)
    print(f"\n  Checking action temporal variation...")
    first_sample_actions = actions[0]  # [H, D]
    variation_per_dim = np.std(first_sample_actions, axis=0)  # Std across time for each dim
    low_variation_dims = np.sum(variation_per_dim < 0.001)
    if low_variation_dims > actions.shape[2] * 0.5:
        print(f"  ⚠️  WARNING: {low_variation_dims}/{actions.shape[2]} dimensions have very low temporal variation!")
        print(f"      This might indicate static/repeated actions")
    else:
        print(f"  ✓ Actions show temporal variation ({low_variation_dims}/{actions.shape[2]} dims static)")
    
    # ========== IMAGES CHECK ==========
    print(f"\n[IMAGES]")
    if hasattr(observation, 'images') and observation.images:
        for img_name, img in observation.images.items():
            print(f"  {img_name}:")
            print(f"    Shape: {img.shape}, Dtype: {img.dtype}")
            print(f"    Range: [{img.min():.1f}, {img.max():.1f}], Mean: {img.mean():.1f}")
            
            # Check first image
            if np.all(img[0] == 0):
                print(f"    ⚠️  First image is all zeros!")
            else:
                print(f"    ✓ First image has non-zero values")

# ============================================================================
# SECTION 4: NORMALIZED DATA INSPECTION
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 4: NORMALIZED DATA INSPECTION (After All Transforms)")
print("=" * 100)

if data_config.norm_stats is None:
    print("\n⚠️  SKIPPING: Cannot check normalized data without norm stats")
    print("   Please compute norm stats first!")
else:
    print("\n⏳ Loading normalized data (first 2 batches)...")
    
    normalized_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=2,
        skip_norm_stats=False,  # Apply normalization
    )
    
    for batch_idx, (observation, actions) in enumerate(normalized_loader):
        print(f"\n{'─'*100}")
        print(f"NORMALIZED BATCH {batch_idx + 1}/2")
        print(f"{'─'*100}")
        
        # ========== STATE (normalized) ==========
        print(f"\n[NORMALIZED STATE]")
        if hasattr(observation, 'state') and observation.state is not None:
            state = observation.state
            print(f"  Shape: {state.shape}")
            print(f"  Range: [{state.min():.3f}, {state.max():.3f}]")
            print(f"  Mean: {state.mean():.3f}, Std: {state.std():.3f}")
            
            # For normalized data, we expect roughly mean≈0, std≈1
            if abs(state.mean()) > 0.5:
                print(f"  ⚠️  WARNING: Normalized state mean is far from 0!")
            if state.std() < 0.5 or state.std() > 2.0:
                print(f"  ⚠️  WARNING: Normalized state std is far from 1!")
            
            # Check for NaN/Inf
            if np.any(np.isnan(state)):
                print(f"  ❌ CRITICAL: State contains NaN values!")
            if np.any(np.isinf(state)):
                print(f"  ❌ CRITICAL: State contains Inf values!")
        
        # ========== ACTIONS (normalized) ==========
        print(f"\n[NORMALIZED ACTIONS]")
        print(f"  Shape: {actions.shape}")
        print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"  Mean: {actions.mean():.3f}, Std: {actions.std():.3f}")
        
        # Check for NaN/Inf
        if np.any(np.isnan(actions)):
            print(f"  ❌ CRITICAL: Actions contain NaN values!")
        if np.any(np.isinf(actions)):
            print(f"  ❌ CRITICAL: Actions contain Inf values!")
        
        # ========== PROMPTS (tokenized) ==========
        print(f"\n[TOKENIZED PROMPTS]")
        if hasattr(observation, 'tokenized_prompt'):
            tokens = observation.tokenized_prompt
            print(f"  Tokens shape: {tokens.shape}")
            print(f"  Token range: [{tokens.min()}, {tokens.max()}]")
            print(f"  First sample tokens (first 30): {tokens[0, :30]}")
            
            if hasattr(observation, 'tokenized_prompt_mask'):
                mask = observation.tokenized_prompt_mask
                active_tokens = mask.sum(axis=1)
                print(f"  Active tokens per sample: min={active_tokens.min()}, "
                      f"max={active_tokens.max()}, mean={active_tokens.mean():.1f}")
        else:
            print(f"  ⚠️  WARNING: No tokenized prompts found!")

# ============================================================================
# SECTION 5: FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 100)
print("SECTION 5: DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
print("=" * 100)

issues_found = []
recommendations = []

# Check 1: Norm stats
if data_config.norm_stats is None:
    issues_found.append("❌ Normalization statistics not loaded")
    recommendations.append(
        "1. Compute norm stats:\n"
        "   uv run --group rlds scripts/compute_norm_stats.py \\\n"
        "       --config-name pi0_lohrbench_rlds_finetune \\\n"
        "       --max-frames 100000"
    )

# Check 2: LR schedule
if (hasattr(lr_schedule, 'decay_steps') and 
    hasattr(lr_schedule, 'decay_lr') and
    lr_schedule.decay_steps < config.num_train_steps):
    issues_found.append(f"❌ LR schedule decays too early ({lr_schedule.decay_steps} < {config.num_train_steps})")
    recommendations.append(
        "2. Fix learning rate schedule in config.py:\n"
        "   lr_schedule=_optimizer.CosineDecaySchedule(\n"
        f"       warmup_steps=2_000,\n"
        f"       peak_lr=5e-5,\n"
        f"       decay_steps={config.num_train_steps},  # Match num_train_steps!\n"
        "       decay_lr=5e-6,\n"
        "   ),"
    )

# Check 3: Action dimensions
# (Would need to actually run the data loader to check this - already done above)

print(f"\n{'='*100}")
if issues_found:
    print("ISSUES FOUND:")
    for issue in issues_found:
        print(f"  {issue}")
    
    print(f"\n{'='*100}")
    print("RECOMMENDED FIXES:")
    for rec in recommendations:
        print(f"\n{rec}")
else:
    print("✅ No major configuration issues detected!")
    print("✅ If loss is still ~0.5, check the data quality and prompt handling above")

print("\n" + "=" * 100)
print("DEBUG SCRIPT COMPLETE")
print("=" * 100)