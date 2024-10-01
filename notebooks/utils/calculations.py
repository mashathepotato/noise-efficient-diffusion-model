from scipy.stats import entropy
import numpy as np
import torch
import math

def calculate_local_entropy(image, patch_size=8):
    """
    Calculate local entropy for non-overlapping patches of an image.
    
    Args:
        image (torch.Tensor): A single image tensor of shape (C, H, W).
        patch_size (int): Size of the square patches to extract.
    
    Returns:
        entropy_map (np.ndarray): A 2D array containing entropy values for each patch.
    """
    image = image.cpu().numpy()
    
    C, H, W = image.shape
    
    num_patches_x = H // patch_size
    num_patches_y = W // patch_size
    
    entropy_map = np.zeros((num_patches_x, num_patches_y))

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            patch = image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            
            patch_data = patch.flatten()
            patch_data = (patch_data - patch_data.min()) / (patch_data.max() - patch_data.min() + 1e-10)  # Adding epsilon
            
            hist, _ = np.histogram(patch_data, bins=256, range=(0, 1), density=True)
            
            patch_entropy = entropy(hist + 1e-10)  # Add a small constant to avoid log(0)
            
            entropy_map[i, j] = patch_entropy
    
    return entropy_map

def calculate_timesteps(max_timesteps, normalized_randomness):
    """
    Calculate timesteps based on normalized randomness, avoiding NaNs and negative values.
    """
    # Ensure normalized randomness is clipped to avoid negative values or large exponents
    if normalized_randomness != normalized_randomness:  # Check for NaN
        normalized_randomness = 0.5  # Set a default value if NaN
    normalized_randomness = max(1e-10, normalized_randomness)  # Avoid log(0) or negative values
    
    return int((max_timesteps-1) / math.exp(normalized_randomness))


def apply_noise_based_on_randomness(image, diffusion, randomness_map, patch_size=8, max_timesteps=1000):
    noisy_image = image.clone()

    num_patches_x, num_patches_y = randomness_map.shape

    min_randomness = randomness_map.min()
    max_randomness = randomness_map.max()
    if max_randomness - min_randomness == 0:
        normalized_randomness = 1
    else:
        normalized_randomness = (randomness_map - min_randomness) / (max_randomness - min_randomness)

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            patch = image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

            # timesteps = int((1 - normalized_randomness[i, j]) * max_timesteps)
            # timesteps = int((max_timesteps-1)/math.exp(normalized_randomness[i, j]))
            timesteps = calculate_timesteps(max_timesteps, normalized_randomness[i, j])
            # print(f"For ra {normalized_randomness[i, j]} noising with {timesteps} timesteps")

            t = torch.tensor([timesteps], device=image.device)
            noisy_patch, _ = diffusion.forward_diffusion(patch.unsqueeze(0), t)

            noisy_image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = noisy_patch.squeeze(0)

    return noisy_image