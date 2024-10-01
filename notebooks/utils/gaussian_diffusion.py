import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianDiffusion(nn.Module):
    def __init__(self, img_size, timesteps=200, beta_start=0.0001, beta_end=0.02, device='cpu'):
        super(GaussianDiffusion, self).__init__()
        self.img_size = img_size
        self.timesteps = timesteps
        self.device = device
        
        # Linear schedule for beta values
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # Precompute alphas and their products
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod).to(device)
    
    def forward_diffusion(self, x_0, t):
        """
        Adds noise to the input data at time step `t`.
        """
        noise = torch.randn_like(x_0).to(self.device)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise
    
    def reverse_diffusion(self, model, x_t, t):
        """
        Removes noise at time step `t` using a denoising model.
        """
        # Predict the noise
        predicted_noise = model(x_t, t)
        
        # Estimate x_0 (the original image)
        sqrt_recip_alpha_t = (1.0 / self.sqrt_alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_recipm1_alpha_t = (1.0 / self.sqrt_one_minus_alphas_cumprod[t]).view(-1, 1, 1, 1)
        x_0_pred = sqrt_recip_alpha_t * x_t - sqrt_recipm1_alpha_t * predicted_noise
        
        # Compute the variance
        variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t).to(self.device)
        x_prev = x_0_pred + torch.sqrt(variance) * noise
        return x_prev

    def sample(self, model, img_shape, n_samples):
        """
        Generates samples by iteratively denoising.
        """
        # Start with random noise
        x_t = torch.randn((n_samples, *img_shape)).to(self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.timesteps)):
            x_t = self.reverse_diffusion(model, x_t, torch.full((n_samples,), t, device=self.device, dtype=torch.long))
        
        return x_t