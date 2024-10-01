import matplotlib.pyplot as plt
import numpy as np

def visualize_noisy_samples(images, noisy_images, num_samples=5):
    """
    Visualize original images and their corresponding noisy versions.
    
    Args:
        images (torch.Tensor): Batch of original images.
        noisy_images (torch.Tensor): Batch of noisy images.
        num_samples (int): Number of samples to visualize.
    """
    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    
    images = (images * 0.5) + 0.5 
    noisy_images = (noisy_images * 0.5) + 0.5 
    
    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        axs[0, i].imshow(img)
        axs[0, i].axis('off')
        axs[0, i].set_title('Original')
        
        noisy_img = np.transpose(noisy_images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        axs[1, i].imshow(noisy_img)
        axs[1, i].axis('off')
        axs[1, i].set_title('Noisy')

    plt.tight_layout()
    plt.show()

def visualize_image_with_patch_entropy(image, entropy_map, patch_size=8):
    """
    Visualize the image with entropy scores for each patch.
    
    Args:
        image (torch.Tensor): The original image tensor.
        entropy_map (np.ndarray): A 2D array of entropy values for each patch.
        patch_size (int): Size of the patches.
    """
    image = image.cpu().numpy()
    image = (image * 0.5) + 0.5  # Undo normalization
    image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

    num_patches_x, num_patches_y = entropy_map.shape

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            y_pos = (i * patch_size) + patch_size // 2
            x_pos = (j * patch_size) + patch_size // 2
            ax.text(x_pos, y_pos, f"{entropy_map[i, j]:.2f}", color='red', 
                    ha='center', va='center', fontsize=10, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    plt.axis('off')
    plt.show()


def visualize_image_with_patches(image, patch_size=8, show_borders=True):
    """
    Visualize the image with an option to display patch borders.

    Args:
        image (torch.Tensor): The input image tensor.
        patch_size (int): Size of the patches.
        show_borders (bool): Whether to show the patch borders or not.
    """
    image = image.cpu().numpy()
    image = (image * 0.5) + 0.5  # Undo normalization
    image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)

    if show_borders:
        H, W, _ = image.shape
        num_patches_x = H // patch_size
        num_patches_y = W // patch_size

        for i in range(1, num_patches_x):
            ax.axhline(i * patch_size, color='white', linestyle='--', linewidth=1)
        for j in range(1, num_patches_y):
            ax.axvline(j * patch_size, color='white', linestyle='--', linewidth=1)

    plt.axis('off')
    plt.show()