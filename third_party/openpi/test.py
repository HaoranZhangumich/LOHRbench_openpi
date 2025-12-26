# Comprehensive dataset verification script
import jax
import numpy as np
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# Load your config
config = _config.get_config("pi0_lohrbench_rlds_finetune")

# Create data loader with just a few batches for testing
data_loader = _data_loader.create_data_loader(
    config,
    shuffle=False,
    num_batches=5,
    skip_norm_stats=True,  # Skip normalization to see raw values
)

print("=" * 80)
print("COMPREHENSIVE DATASET VERIFICATION")
print("=" * 80)

all_issues = []

# Iterate through batches
for i, (observation, actions) in enumerate(data_loader):
    print(f"\n{'='*80}")
    print(f"BATCH {i+1}")
    print(f"{'='*80}")
    
    batch_issues = []
    
    # ========== Check Images ==========
    print("\n[IMAGES]")
    if hasattr(observation, 'images') and observation.images:
        for img_name, img in observation.images.items():
            print(f"  {img_name}:")
            print(f"    Shape: {img.shape}, Dtype: {img.dtype}")
            print(f"    Range: [{img.min():.2f}, {img.max():.2f}], Mean: {img.mean():.2f}")
            
            # Check for NaN/Inf
            if np.any(np.isnan(img)):
                issue = f"    ⚠️  WARNING: {img_name} contains NaN values!"
                print(issue)
                batch_issues.append(issue)
            if np.any(np.isinf(img)):
                issue = f"    ⚠️  WARNING: {img_name} contains Inf values!"
                print(issue)
                batch_issues.append(issue)
            
            # Check for all-zero images (padding)
            num_zero_images = np.sum(np.all(img == 0, axis=(1, 2, 3)))
            if num_zero_images > 0:
                issue = f"    ⚠️  WARNING: {num_zero_images}/{img.shape[0]} images are all zeros (padding)!"
                print(issue)
                batch_issues.append(issue)
            
            # Check for suspiciously low variance (might be corrupted/padding)
            variances = np.var(img, axis=(1, 2, 3))
            low_var_count = np.sum(variances < 0.01)
            if low_var_count > img.shape[0] * 0.1:  # More than 10% have low variance
                issue = f"    ⚠️  WARNING: {low_var_count}/{img.shape[0]} images have very low variance (possibly corrupted)!"
                print(issue)
                batch_issues.append(issue)
            
            print(f"    ✓ No padding/empty images detected" if num_zero_images == 0 else "")
    
    # ========== Check Image Masks ==========
    print("\n[IMAGE MASKS]")
    if hasattr(observation, 'image_masks') and observation.image_masks:
        for mask_name, mask in observation.image_masks.items():
            mask_array = np.asarray(mask)
            true_count = np.sum(mask_array == True)
            false_count = np.sum(mask_array == False)
            print(f"  {mask_name}: {true_count} real, {false_count} padding")
            
            if false_count > 0:
                issue = f"    ⚠️  WARNING: {mask_name} has {false_count} padded images!"
                print(issue)
                batch_issues.append(issue)
    
    # ========== Check State ==========
    print("\n[STATE]")
    if hasattr(observation, 'state') and observation.state is not None:
        state = observation.state
        print(f"  Shape: {state.shape}, Dtype: {state.dtype}")
        print(f"  Range: [{state.min():.3f}, {state.max():.3f}], Mean: {state.mean():.3f}")
        
        # Check for NaN/Inf
        if np.any(np.isnan(state)):
            issue = f"  ⚠️  WARNING: State contains NaN values!"
            print(issue)
            batch_issues.append(issue)
        if np.any(np.isinf(state)):
            issue = f"  ⚠️  WARNING: State contains Inf values!"
            print(issue)
            batch_issues.append(issue)
        
        # Check for all-zero states (padding)
        num_zero_states = np.sum(np.all(state == 0, axis=1))
        if num_zero_states > 0:
            issue = f"  ⚠️  WARNING: {num_zero_states}/{state.shape[0]} states are all zeros (padding)!"
            print(issue)
            batch_issues.append(issue)
        
        # Check individual dimensions
        print(f"  Per-dimension stats:")
        for dim in range(min(state.shape[1], 8)):  # Show first 8 dims
            dim_values = state[:, dim]
            num_zeros = np.sum(dim_values == 0)
            print(f"    Dim {dim}: range=[{dim_values.min():.3f}, {dim_values.max():.3f}], "
                  f"std={dim_values.std():.3f}, zeros={num_zeros}/{len(dim_values)}")
        
        print(f"  ✓ No padding/empty states detected" if num_zero_states == 0 else "")
    
    # ========== Check Actions ==========
    print("\n[ACTIONS]")
    print(f"  Shape: {actions.shape}, Dtype: {actions.dtype}")
    print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}], Mean: {actions.mean():.3f}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(actions)):
        issue = f"  ⚠️  WARNING: Actions contain NaN values!"
        print(issue)
        batch_issues.append(issue)
    if np.any(np.isinf(actions)):
        issue = f"  ⚠️  WARNING: Actions contain Inf values!"
        print(issue)
        batch_issues.append(issue)
    
    # Check for all-zero action sequences (padding)
    num_zero_actions = np.sum(np.all(actions == 0, axis=(1, 2)))
    if num_zero_actions > 0:
        issue = f"  ⚠️  WARNING: {num_zero_actions}/{actions.shape[0]} action sequences are all zeros!"
        print(issue)
        batch_issues.append(issue)
    
    # Check per action dimension
    print(f"  Per-dimension stats (first timestep):")
    for dim in range(actions.shape[2]):
        dim_values = actions[:, 0, dim]  # First timestep
        num_zeros = np.sum(dim_values == 0)
        print(f"    Dim {dim}: range=[{dim_values.min():.3f}, {dim_values.max():.3f}], "
              f"std={dim_values.std():.3f}, zeros={num_zeros}/{len(dim_values)}")
    
    print(f"  ✓ No padding/empty actions detected" if num_zero_actions == 0 else "")
    
    # ========== Check Prompts ==========
    print("\n[PROMPTS]")
    # Check for raw prompts (before tokenization)
    if hasattr(observation, 'prompt') and observation.prompt is not None:
        prompts = observation.prompt
        print(f"  Raw prompts found: {type(prompts)}")
        if isinstance(prompts, list):
            print(f"  Count: {len(prompts)} prompts")
            print(f"  Sample prompts (first 3): {prompts[:3]}")
    # Check for tokenized prompts (after model transforms)
    elif hasattr(observation, 'tokenized_prompt') and observation.tokenized_prompt is not None:
        tokenized = observation.tokenized_prompt
        print(f"  ✓ Prompts have been tokenized")
        print(f"  Tokenized prompt shape: {tokenized.shape}, dtype: {tokenized.dtype}")
        print(f"  Token range: [{tokenized.min()}, {tokenized.max()}]")
        print(f"  Sample tokens (first sample, first 20): {tokenized[0, :20]}")
        
        # Check for prompt mask
        if hasattr(observation, 'tokenized_prompt_mask'):
            mask = observation.tokenized_prompt_mask
            print(f"  Prompt mask shape: {mask.shape}")
            print(f"  Active tokens per sample (mean): {mask.sum(axis=1).mean():.1f}")
    else:
        issue = f"  ⚠️  WARNING: No prompts (raw or tokenized) found in observation!"
        print(issue)
        batch_issues.append(issue)
        
        # ========== Batch Summary ==========
        if batch_issues:
            print(f"\n{'!'*80}")
            print(f"BATCH {i+1} ISSUES SUMMARY: {len(batch_issues)} issues found")
            print(f"{'!'*80}")
            for issue in batch_issues:
                print(issue)
            all_issues.extend(batch_issues)
        else:
            print(f"\n{'✓'*80}")
            print(f"BATCH {i+1}: ALL CHECKS PASSED - No issues detected!")
            print(f"{'✓'*80}")

# ========== Final Summary ==========
print("\n" + "=" * 80)
print("FINAL VERIFICATION SUMMARY")
print("=" * 80)

if all_issues:
    print(f"\n❌ FOUND {len(all_issues)} TOTAL ISSUES:")
    for idx, issue in enumerate(all_issues, 1):
        print(f"{idx}. {issue}")
    print("\n⚠️  Please review and fix these issues before training!")
else:
    print("\n✅ ALL CHECKS PASSED!")
    print("✅ Dataset is clean - no padding, empty values, or NaN/Inf detected")
    print("✅ Ready for training!")

print("\n" + "=" * 80)