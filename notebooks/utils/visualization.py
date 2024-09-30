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
    
    # Denormalize the images (assuming they were normalized between -1 and 1)
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